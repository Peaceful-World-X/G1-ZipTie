"""Video-reference order: right tail pickup, left head clamp, flip, insert, place."""
from simulation import G1ZipTieEnv as InsertionEnv, ROOT, smooth
import numpy as np
import mujoco
import json
from scipy.spatial.transform import Rotation, Slerp
from reference_motion import ReferenceMotion
from vertical_handoff import targets as sequence_targets,tail_pose,regrasp_rotation

DURATION=24.
SCRIPT_RATE=1.
PHASES=[(0,'PICK_RIGHT','右手夹带尾'),(3,'HANDOFF','横向提起 · 递交锁扣'),(5.8,'CLAMP_LEFT','左手接住锁扣'),(6.5,'RELEASE_REORIENT','左爪横持 · 带尾向下'),(8.2,'VERTICAL_HOLD','左爪横持 · 带尾向下'),(8.8,'REGRASP_RIGHT','右手重新夹住带尾'),(9.5,'FLIP_RIGHT','右腕回翻 · 带尾上送'),(12.6,'INSERT_RIGHT','孔面竖直 · 侧向穿入'),(19,'PLACE_LEFT','右手松开 · 左手放下'),(21.8,'RETREAT','双手撤离'),(23.4,'COMPLETE','完成')]

def mix_pose(a,b,w):
    w=float(np.clip(smooth(w),0,1))
    rot=Slerp([0,1],Rotation.from_matrix(np.stack([a[1],b[1]])))(w).as_matrix()
    return (a[0]*(1-w)+b[0]*w,rot)

class G1ZipTieEnv(InsertionEnv):
    def __init__(self):
        self.layout=json.loads((ROOT/'layout.json').read_text(encoding='utf-8'))
        self.ik_sites={'left':'left_grip','right':'right_grip'}
        if not (ROOT/'scenes/workcell_sequence.xml').exists():
            from build_sequence import build_sequence
            build_sequence()
        super().__init__(ROOT/'scenes/workcell_sequence.xml')

    def pose(self,site):
        s=self.data.site(site)
        return s.xpos.copy(),s.xmat.reshape(3,3).copy()

    def reset(self,seed=42):
        m=self.model; d=self.data
        self.reference=ReferenceMotion(m)
        self.right_regrasped=False
        mujoco.mj_resetData(m,d)
        for s in ('head_grasp','tail_grasp'):
            m.site_quat[m.site(s).id]=[1,0,0,0]
        self.success=False; self.ever_entered=False; self.hold=0.; self.max_tail_contact=0.;self.max_strip_gap=0.
        self.pulse_until=-1; self.events=[]; self.grasped={'left':False,'right':False}; self.offsets={}; self.saved={}; self.inserted=False; self.placed=False
        self.plan_errors=[]; self.plan=[]; self.last_ik=-1; self.failure=None
        self.collision_peaks={}
        self.action_log=[]
        for side in ('left','right'):
            for n,v in [('hip_pitch',-.08),('knee',.16),('ankle_pitch',-.08),('elbow',1.)]:
                d.qpos[m.joint(f'{side}_{n}_joint').qposadr[0]]=v
            for i in (1,2):d.qpos[m.joint(f'{side}_jaw_{i}').qposadr[0]]=.025
        mujoco.mj_forward(m,d)
        reference=np.asarray(self.layout['initial_arm_reference'])
        mirror=np.array([1,-1,-1,1,-1,1,-1])
        left=(reference[:7]+reference[7:]*mirror)/2
        d.qpos[self.arms['left']]=left;d.qpos[self.arms['right']]=left*mirror
        self.command=d.qpos.copy(); self.desired=self.command.copy()
        d.ctrl[:]=d.qpos[self.ctrl_q]
        mujoco.mj_forward(m,d)
        self.home={'left':self.pose('left_grip'),'right':self.pose('right_grip')}
        self.home_joints={side:d.qpos[self.arms[side]].copy() for side in ('left','right')}
        return self.observe(),self.info()

    def phase(self):
        return next(p for p in reversed(PHASES) if self.data.time>=p[0])

    def update_success(self):
        super().update_success()
        for c in self.data.contact:
            if c.dist>=0 or not ((self.model.geom_contype[c.geom1]|self.model.geom_contype[c.geom2])&16):continue
            labels=[self.model.body(self.model.geom_bodyid[g]).name or self.model.geom(g).name or 'world' for g in (c.geom1,c.geom2)]
            key=' / '.join(sorted(labels))
            self.collision_peaks[key]=max(self.collision_peaks.get(key,0),float(-c.dist*1000))

    def capture(self,side):
        obj='tail' if side=='right' else 'head'; site=obj+'_grasp'
        pg,Rg=self.pose(side+'_grip'); po,Ro=self.pose(site)
        distance=np.linalg.norm(pg-po)
        if distance>.006: return False
        # Capture the existing orientation at contact, never reposition the object.
        body='tail_body' if side=='right' else 'lock_head'
        Rbody=self.data.body(body).xmat.reshape(3,3).copy()
        self.model.site_quat[self.model.site(site).id]=Rotation.from_matrix(Rbody.T@Rg).as_quat()[[3,0,1,2]]
        self.data.eq_active[self.model.equality(side+'_grasp').id]=1
        self.grasped[side]=True
        relative_rotation=Rg.T@Rbody
        # Site welds coincide at their anchors; do not retain the pre-capture
        # approach gap as a permanent object/gripper translation offset.
        self.offsets[side]=(-relative_rotation@self.model.site(site).pos,relative_rotation)
        self.saved[side+'_jaw_closed']=float(np.clip(abs(relative_rotation[1,1])*.0015+abs(relative_rotation[1,2])*.0004+.00005,.00045,.0018))
        if side=='left':self.saved['left_hold_joints']=self.data.qpos[self.arms['left']].copy()
        again=side=='right' and 'right_capture_time' in self.saved
        if again:self.right_regrasped=True
        self.saved[side+'_capture_time']=self.data.time
        self.saved[side+'_capture_quat']=self.model.site_quat[self.model.site(site).id].copy()
        self.events.append({'event':'right_regrasp' if again else side+'_grasp','time':self.data.time,'distance_mm':float(distance*1000)})
        return True

    def object_to_grip(self,side,pose):
        offset,Rrel=self.offsets[side]
        Rg=pose[1]@Rrel.T
        return pose[0]-Rg@offset,Rg

    def latch(self):
        a=self.data.body('lock_head'); b=self.data.body('tail_body')
        R=a.xmat.reshape(3,3); p=R.T@(b.xpos-a.xpos); rot=R.T@b.xmat.reshape(3,3)
        eid=self.model.equality('thread_latch').id
        self.model.eq_data[eid,3:6]=p
        self.model.eq_data[eid,6:10]=Rotation.from_matrix(rot).as_quat()[[3,0,1,2]]
        self.data.eq_active[eid]=1; self.inserted=True
        self.events.append({'event':'verified_insertion_then_latch','time':self.data.time,**self.metrics()})

    def release(self,side,event):
        self.data.eq_active[self.model.equality(side+'_grasp').id]=0
        self.grasped[side]=False
        self.events.append({'event':event,'time':self.data.time})

    def targets(self):
        return sequence_targets(self)

    def solve_ik(self,side,pos,rot,seed,iterations=160):
        # Keep the right elbow outside the torso during regrasp/insertion.
        jid=self.model.joint('right_shoulder_roll_joint').id
        upper=self.model.jnt_range[jid,1]
        try:
            if side=='right' and 6.5<=self.data.time<19:self.model.jnt_range[jid,1]=min(upper,.18)
            return super().solve_ik(side,pos,rot,seed,iterations)
        finally:self.model.jnt_range[jid,1]=upper

    def step(self,action=None,dt=.02):
        for _ in range(max(1,round(dt/self.model.opt.timestep))):
            if self.data.time-self.last_ik>=.04-1e-9:
                poses,jaws=self.targets()
                if self.failure:return self.observe(),0.,False,True,self.info()
                q=self.desired.copy()
                reorienting=6.5<=self.data.time<8.6
                if 6.5<=self.data.time<8.2 and 'left_vertical_end_joints' not in self.saved:
                    self.saved['left_vertical_start_joints']=self.command.copy()
                    end,error=self.solve_ik('left',*poses['left'],q,iterations=160)
                    if error>.001:self.failure='Vertical left hold is unreachable'
                    self.saved['left_vertical_end_joints']=end
                if reorienting and 'regrasp_end_joints' not in self.saved:
                    self.saved['regrasp_start_joints']=self.command.copy()
                    hp,H=self.saved['head_goal']
                    relative=np.array(self.layout['strip_initial_endpoint'])+self.model.site('tail_grasp').pos-self.model.site('tail_root').pos
                    endpoint=hp+H@relative
                    seed=q.copy();seed[self.arms['right']]=self.reference.joints(12,'right')
                    end,error=self.solve_ik('right',endpoint,regrasp_rotation(H),seed,iterations=120)
                    if error>.002:self.failure='Regrasp posture is unreachable'
                    self.saved['regrasp_end_joints']=end
                    # Roll the camera over the source tray first. Only descend
                    # after the inverted housing is inside the clear center lane.
                    hover,error=self.solve_ik('right',endpoint+np.array([0,0,.085]),
                                              regrasp_rotation(H),end,iterations=120)
                    if error>.002:self.failure='Inverted wrist clearance pose is unreachable'
                    self.saved['regrasp_hover_joints']=hover
                    # Follow the descent on the same IK branch as the hover.
                    # The two endpoint solutions can otherwise fold the arm
                    # through the tray even when both endpoints are clear.
                    tip_joints=hover.copy()
                    for lift in np.linspace(.085,0.,10)[1:]:
                        tip_joints,error=self.solve_ik('right',endpoint+np.array([0,0,lift]),
                                                       regrasp_rotation(H),tip_joints,iterations=120)
                    self.saved['regrasp_end_joints']=tip_joints
                    if error>.002:self.failure='Continuous descent to tail is unreachable'
                    self.saved['regrasp_goal_updated']=self.data.time
                if reorienting and self.data.time>=8.1 and self.data.time-self.saved['regrasp_goal_updated']>=.15:
                    # Refine the approach after the freely held strip settles.
                    # Keeping the pre-release prediction can push the elbow
                    # into the torso even though the settled grasp is clear.
                    point=self.pose('tail_grasp')[0]
                    rotation=regrasp_rotation(self.data.body('tail_body').xmat.reshape(3,3))
                    prior=self.saved['regrasp_end_joints']
                    end,error=self.solve_ik('right',point,rotation,prior,iterations=120)
                    if error<.001:
                        ids=self.arms['right'];end[ids]=prior[ids]+np.clip(end[ids]-prior[ids],-.12,.12)
                        self.saved['regrasp_end_joints']=end
                    self.saved['regrasp_goal_updated']=self.data.time
                for side in ('left','right'):
                    release=19 if side=='right' else 21.8
                    script_t=self.data.time*SCRIPT_RATE
                    if side=='left' and 6.5<=script_t<8.2:
                        w=float(smooth((script_t-6.5)/1.5));ids=self.arms[side]
                        q[ids]=self.saved['left_vertical_start_joints'][ids]*(1-w)+self.saved['left_vertical_end_joints'][ids]*w;e=0.
                    elif side=='left' and self.grasped['left'] and script_t<19:
                        q[self.arms[side]]=self.saved['left_hold_joints'];e=0.
                    elif side=='right' and reorienting:
                        ids=self.arms[side]
                        if script_t < 7.82:
                            start=self.saved['regrasp_start_joints']
                            end=self.saved['regrasp_hover_joints']
                            w=float(smooth((script_t-6.5)/1.32))
                        else:
                            start=self.saved['regrasp_hover_joints']
                            end=self.saved['regrasp_end_joints']
                            w=float(smooth((script_t-7.82)/.78))
                        q[ids]=start[ids]*(1-w)+end[ids]*w
                        e=0.
                    elif script_t>=release and self.inserted:
                        start=self.saved.setdefault(side+'_retreat_joints',self.command[self.arms[side]].copy())
                        weight=float(smooth((script_t-release)/1.6))
                        q[self.arms[side]]=start*(1-weight)+self.home_joints[side]*weight
                        e=0.
                    else:
                        # Warm-start from the preceding solution. Restarting
                        # from each recorded sample can switch IK branches.
                        reference=self.reference.joints(5 if side=='left' and self.data.time<6.5 else self.reference.mapped_time(self.data.time),side)
                        if side!='right' or script_t<6.5 or script_t>=19:
                            q[self.arms[side]]+=np.clip(.08*(reference-q[self.arms[side]]),-.04,.04)
                        q,e=self.solve_ik(side,*poses[side],q,iterations=60)
                    self.plan_errors.append({'time':self.data.time,'side':side,'error_m':e})
                    for i in (1,2): q[self.model.joint(f'{side}_jaw_{i}').qposadr[0]]=jaws[side]
                self.desired=q;self.last_ik=self.data.time
                self.action_log.append((self.data.time,q[self.ctrl_q].copy(),self.data.eq_active.copy(),self.model.site_quat.copy(),self.model.eq_data.copy()))
            rate=2.8*self.model.opt.timestep
            delta=np.clip(self.desired[self.ctrl_q]-self.command[self.ctrl_q],-rate,rate)
            self.command[self.ctrl_q]+=delta
            self.data.ctrl[:]=self.command[self.ctrl_q]
            self.data.qfrc_applied[:]=0;self.data.qfrc_applied[self.ctrl_q]=self.data.qfrc_bias[self.ctrl_q]
            self.data.xfrc_applied[:]=0
            if self.data.time<self.pulse_until:self.data.xfrc_applied[self.model.body('strip_16').id,0]=.003
            prior_time=self.data.time
            mujoco.mj_step(self.model,self.data);mujoco.mj_forward(self.model,self.data)
            if self.data.time<prior_time: raise RuntimeError('MuJoCo reset due to numerical instability')
            self.update_success()
        return self.observe(),float(self.inserted and self.placed),self.placed,self.data.time>=DURATION,self.info()

    def info(self):
        info=super().info()
        p=self.data.body('lock_head').xpos
        cfg=self.layout['destination_tray'];center=np.array(cfg['center']);size=np.array(cfg['half_size'])
        in_tray=bool(np.all(np.abs(p[:2]-center[:2])<size[:2]) and center[2]-.02<p[2]<center[2]+.1)
        info.update(task_complete=bool(self.inserted and self.placed and self.data.time>=23.3 and in_tray and self.failure is None),inserted=self.inserted,placed=self.placed,head_in_destination=in_tray,failure=self.failure)
        info['physics_scope']='official Dex1-1; free elastic tie; proximity-triggered grasp welds; verified insertion; simplified latch; gravity placement'
        info['max_robot_penetration_mm']=max(self.collision_peaks.values(),default=0.)
        return info

if __name__=='__main__':
    import json,time
    e=G1ZipTieEnv();last=-1
    while e.data.time<DURATION:
        e.step(dt=.02)
        if int(e.data.time)//2!=last:
            last=int(e.data.time)//2
            print(round(e.data.time,2),e.phase()[1],e.grasped,e.failure,e.metrics(),e.plan_errors[-2:],flush=True)
            from simulation import render
            from PIL import Image
            Image.fromarray(render(e,'hands',width=1000,height=700)).save(ROOT/'recordings'/f'planning_{last*2:02}.png')
        if e.failure:break
    print(json.dumps(e.events,indent=2),flush=True)
    print(json.dumps(e.info(),indent=2),flush=True)
    if e.failure:
        np.savez_compressed(ROOT/'recordings/planner_failure_state.npz',qpos=e.data.qpos,qvel=e.data.qvel,ctrl=e.data.ctrl,sitequat=e.model.site_quat,eqdata=e.model.eq_data,eqactive=e.data.eq_active,time=e.data.time)
        from simulation import render
        from PIL import Image
        Image.fromarray(render(e,'hands')).save(ROOT/'recordings/planner_failure.png')
    # Always retain diagnostics, including unsuccessful planning attempts.
    (ROOT/'recordings/planner_diagnostics.json').write_text(json.dumps({'events':e.events,'final':e.info(),'collision_peaks_mm':e.collision_peaks,'last_ik':e.plan_errors[-2:]},indent=2),encoding='utf-8')
    if e.inserted and e.placed:
        import hashlib
        np.savez_compressed(ROOT/'recordings/actions.npz',times=np.array([a[0] for a in e.action_log]),ctrl=np.array([a[1] for a in e.action_log]),eqactive=np.array([a[2] for a in e.action_log]),sitequat=np.array([a[3] for a in e.action_log]),eqdata=np.array([a[4] for a in e.action_log]),scene_hash=hashlib.sha256((ROOT/'scenes/workcell_sequence.xml').read_bytes()).hexdigest(),layout_hash=hashlib.sha256((ROOT/'layout.json').read_bytes()).hexdigest())
        (ROOT/'recordings/sequence_verification.json').write_text(json.dumps({'events':e.events,'final':e.info(),'collision_peaks_mm':e.collision_peaks,'warnings':int(e.data.warning.number.sum())},indent=2),encoding='utf-8')

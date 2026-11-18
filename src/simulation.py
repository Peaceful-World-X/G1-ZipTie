"""Joint-actuated G1 insertion demo. No per-step teleportation of live bodies."""
from pathlib import Path
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
DURATION=18.0
PHASES=[(0,'READY','预夹持 · 准备'),(2,'APPROACH','双臂接近'),(6,'ALIGN','带尾对准锁扣'),(9,'INSERT','穿过锁扣通道'),(13,'VERIFY','保持 · 验证穿入'),(16,'COMPLETE','完成展示')]

def smooth(a):
    a=np.clip(a,0,1)
    return np.clip(a*a*a*(10+a*(-15+6*a)),0,1)

class G1ZipTieEnv:
    """reset / step / render API; step accepts left_arm/right_arm joint targets.

    Official Dex1-1 tools pre-clamp the lock and rigid tail. The 32-link ribbon
    has physical mass, elastic bending, twist and damping; no ratchet yet.
    """
    def __init__(self,scene_path=None):
        if not (ROOT/'scenes/workcell.xml').exists():
            from build_scene import build
            build()
        self.model=mujoco.MjModel.from_xml_path(str(scene_path or ROOT/'scenes/workcell.xml'))
        self.data=mujoco.MjData(self.model)
        self.ikdata=mujoco.MjData(self.model)
        m=self.model
        self.arms={}
        for side in ('left','right'):
            names=[f'{side}_{n}_joint' for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]
            self.arms[side]=np.array([m.joint(n).qposadr[0] for n in names])
        self.ctrl_q=np.array([m.jnt_qposadr[m.actuator_trnid[i,0]] for i in range(m.nu)])
        self.strip_q=np.array([m.joint(f'strip_{i:02}_{a}').qposadr[0] for i in range(32) for a in ('twist','bend','side')])
        self.strip_j=np.array([m.joint(f'strip_{i:02}_{a}').id for i in range(32) for a in ('twist','bend','side')])
        self.strip_sites=np.array([m.site(f'strip_site_{i:02}').id for i in range(32)])
        self.pulse_until=-1
        self.hole=m.site('hole').id; self.tip=m.site('tip').id
        self.tailgeom=m.geom('rigid_tail').id
        self.lockgeoms={m.geom('lock_'+s).id for s in ('top','bottom','a','b')}
        self.renderer=None
        self.plan=[]
        self.reset()

    def solve_ik(self, side, pos, rot, seed, iterations=160):
        m=self.model; d=self.ikdata; ids=self.arms[side]
        d.qpos[:]=seed
        sid=m.site(getattr(self,'ik_sites',{}).get(side,'hole' if side=='left' else 'tip')).id
        jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv))
        for _ in range(iterations):
            mujoco.mj_kinematics(m,d); mujoco.mj_comPos(m,d)
            errp=np.asarray(pos)-d.site_xpos[sid]
            erra=Rotation.from_matrix(rot@d.site_xmat[sid].reshape(3,3).T).as_rotvec()
            if np.linalg.norm(errp)<.00003 and np.linalg.norm(erra)<.0003:
                break
            mujoco.mj_jacSite(m,d,jp,jr,sid)
            J=np.vstack((jp[:,ids],.25*jr[:,ids]))
            err=np.r_[errp,.25*erra]
            dq=J.T@np.linalg.solve(J@J.T+.00005*np.eye(6),err)
            dq=np.clip(dq,-.12,.12)
            d.qpos[ids]+=dq
            for q in ids:
                d.qpos[q]=np.clip(d.qpos[q],m.jnt_range[q,0]+.005,m.jnt_range[q,1]-.005)
        mujoco.mj_kinematics(m,d); mujoco.mj_comPos(m,d)
        ep=np.linalg.norm(np.asarray(pos)-d.site_xpos[sid])
        ea=np.linalg.norm(Rotation.from_matrix(rot@d.site_xmat[sid].reshape(3,3).T).as_rotvec())
        if ep>.001 or ea>.01:
            lo=m.jnt_range[ids,0]+.006; hi=m.jnt_range[ids,1]-.006
            rng=np.random.default_rng(42)
            def residual(x):
                d.qpos[ids]=x
                mujoco.mj_kinematics(m,d); mujoco.mj_comPos(m,d)
                return np.r_[d.site_xpos[sid]-pos,.25*Rotation.from_matrix(rot@d.site_xmat[sid].reshape(3,3).T).as_rotvec()]
            best=None
            for trial in range(10):
                x=np.clip(d.qpos[ids] if trial==0 else rng.uniform(lo,hi),lo,hi)
                result=least_squares(residual,x,bounds=(lo,hi),max_nfev=180,ftol=1e-9,xtol=1e-9,gtol=1e-9)
                if best is None or np.linalg.norm(result.fun)<np.linalg.norm(best.fun): best=result
                if np.linalg.norm(best.fun)<.0001: break
            residual(best.x)
        return d.qpos.copy(),float(np.linalg.norm(np.asarray(pos)-d.site_xpos[sid]))

    def reset(self,seed=42):
        mujoco.mj_resetData(self.model,self.data)
        self.success=False; self.hold=0.; self.ever_entered=False
        self.max_tail_contact=0.; self.max_strip_gap=0.; self.records=[]
        self.pulse_until=-1
        # Slight knee flexion produces a natural standing pose on the fixture.
        for side in ('left','right'):
            for n,v in [('hip_pitch',-.08),('knee',.16),('ankle_pitch',-.08),('elbow',1.0)]:
                self.data.qpos[self.model.joint(f'{side}_{n}_joint').qposadr[0]]=v
            for i in (1,2):
                self.data.qpos[self.model.joint(f'{side}_jaw_{i}').qposadr[0]]=.004 if side=='left' else .0015
        if not self.plan:
            q=self.data.qpos.copy()
            self.plan_errors=[]
            # Tool x axes point inward; the hole's +x entrance faces the right hand.
            rl=Rotation.from_euler('z',-90,degrees=True).as_matrix()
            rr=Rotation.from_euler('z',90,degrees=True).as_matrix()
            knots=[(0,[.28,.075,1.02],[.25,-.07,.99]),
                   (2,[.28,.075,1.02],[.25,-.07,.99]),
                   (6,[.32,.035,1.02],[.32,-.04,.99]),
                   (9,[.32,.035,1.02],[.32,-.010,1.02]),
                   (13,[.32,.035,1.02],[.32,.048,1.02]),
                   (18,[.32,.035,1.02],[.32,.048,1.02])]
            q,le=self.solve_ik('left',knots[0][1],rl,q,iterations=600)
            q[self.arms['right']]=q[self.arms['left']]*[1,-1,-1,1,-1,1,-1]
            for t in np.linspace(0,DURATION,181):
                for (ta,la,ra),(tb,lb,rb) in zip(knots,knots[1:]):
                    if t<=tb:
                        w=smooth((t-ta)/(tb-ta))
                        lp=np.array(la)+(np.array(lb)-la)*w
                        rp=np.array(ra)+(np.array(rb)-ra)*w
                        break
                q,le=self.solve_ik('left',lp,rl,q,iterations=300)
                q,re=self.solve_ik('right',rp,rr,q,iterations=300)
                self.plan.append((t,q.copy()))
                self.plan_errors.append({'time':t,'left_m':le,'right_m':re})
            worst=max(max(e['left_m'],e['right_m']) for e in self.plan_errors)
            if worst>.002:
                raise RuntimeError(f'IK plan is outside tolerance: {worst:.4f}m')
        self.data.qpos[:]=self.plan[0][1]
        self.data.ctrl[:]=self.data.qpos[self.ctrl_q]
        mujoco.mj_forward(self.model,self.data)
        return self.observe(),self.info()

    def target(self,t):
        for (ta,qa),(tb,qb) in zip(self.plan,self.plan[1:]):
            if t<=tb:
                return qa+(qb-qa)*np.clip((t-ta)/(tb-ta),0,1)
        return self.plan[-1][1].copy()

    def phase(self):
        return next(p for p in reversed(PHASES) if self.data.time>=p[0])

    def metrics(self):
        d=self.data
        R=d.site_xmat[self.hole].reshape(3,3)
        local=R.T@(d.site_xpos[self.tip]-d.site_xpos[self.hole])
        axis=d.site_xmat[self.tip].reshape(3,3)[:,0]
        angle=np.degrees(np.arccos(np.clip(-R[:,0]@axis,-1,1)))
        # Hole entrance x=+.004, exit=-.004; success requires 5mm beyond exit.
        return dict(depth_mm=float((.004-local[0])*1000),
                    lateral_mm=float(np.linalg.norm(local[1:])*1000),
                    angle_deg=float(angle),local_tip=local.tolist())

    def update_success(self):
        self.max_strip_gap=max(getattr(self,'max_strip_gap',0.),float(np.linalg.norm(self.data.site('strip_end').xpos-self.data.site('tail_root').xpos)))
        v=self.metrics(); x,y,z=v['local_tip']
        if x>.005 and abs(y)<.001 and abs(z)<.0007:
            self.ever_entered=True
        aligned=abs(y)<.0009 and abs(z)<.0007 and v['angle_deg']<3
        valid=self.ever_entered and x<-.009 and aligned
        self.hold=self.hold+self.model.opt.timestep if valid else 0.
        # Current, continuously held measurement, not a timed success animation.
        self.success=self.hold>=.5
        for c in self.data.contact:
            if self.tailgeom in (c.geom1,c.geom2) and ({c.geom1,c.geom2}&self.lockgeoms):
                self.max_tail_contact=max(self.max_tail_contact,float(-c.dist))

    def step(self,action=None,dt=1/60):
        n=max(1,round(dt/self.model.opt.timestep))
        for _ in range(n):
            q=self.target(self.data.time)
            if action:
                for side in ('left','right'):
                    if side+'_arm' in action:
                        vals=np.asarray(action[side+'_arm'],float)
                        if vals.shape!=(7,) or not np.isfinite(vals).all():
                            raise ValueError('Each arm action must contain 7 finite joint targets')
                        ids=self.arms[side]
                        q[ids]=np.clip(vals,self.model.jnt_range[ids,0],self.model.jnt_range[ids,1])
            self.data.ctrl[:]=q[self.ctrl_q]
            # Ideal gravity compensation, with actuator-driven position tracking.
            self.data.qfrc_applied[:]=0
            self.data.qfrc_applied[self.ctrl_q]=self.data.qfrc_bias[self.ctrl_q]
            self.data.xfrc_applied[:]=0
            if self.data.time<self.pulse_until:
                self.data.xfrc_applied[self.model.body('strip_16').id,0]=.003
            mujoco.mj_step(self.model,self.data)
            mujoco.mj_forward(self.model,self.data)
            self.update_success()
        return self.observe(),float(self.success),self.success,self.data.time>=DURATION,self.info()

    def observe(self):
        return {'qpos':self.data.qpos.copy(),'qvel':self.data.qvel.copy(),
                'left_ee':self.data.site_xpos[self.hole].copy(),
                'right_ee':self.data.site_xpos[self.tip].copy()}

    def info(self):
        return {**self.metrics(),'success':self.success,'hold_seconds':self.hold,
                'phase':self.phase()[1],'time':self.data.time,
                'physics_scope':'fixed base; official Dex1-1; pre-clamped rigid tip; 32-link elastic strip; no ratchet',
                'strip_elastic_energy_mJ':float(.5*np.sum(self.model.jnt_stiffness[self.strip_j]*self.data.qpos[self.strip_q]**2)*1000),
                'strip_endpoint_error_mm':float(np.linalg.norm(self.data.site('strip_end').xpos-self.data.site('tail_root').xpos)*1000),
                'max_strip_endpoint_error_mm':getattr(self,'max_strip_gap',0.)*1000,
                'max_tail_penetration_mm':self.max_tail_contact*1000}

    def perturb_strip(self):
        self.pulse_until=self.data.time+.12

    def move_ee(self,hand,position,orientation):
        """Return a joint action for a WORLD-frame target; quaternion order wxyz."""
        if hand not in self.arms: raise ValueError('hand must be left or right')
        quat=np.asarray(orientation)
        rot=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
        q,e=self.solve_ik(hand,position,rot,self.data.qpos)
        if e>.003: raise ValueError(f'Unreachable target: IK residual {e:.4f}m')
        return {hand+'_arm':q[self.arms[hand]].copy()}

    def close(self):
        if self.renderer: self.renderer.close()
        if getattr(self,'sensor_renderer',None):self.sensor_renderer.close()

def projected_box_gap(env, first, second, axis):
    """Lower bound on separation along a unit world axis for two MuJoCo boxes."""
    a, b = env.data.geom(first), env.data.geom(second)
    radius_a = np.abs(axis @ a.xmat.reshape(3, 3)) @ env.model.geom_size[a.id]
    radius_b = np.abs(axis @ b.xmat.reshape(3, 3)) @ env.model.geom_size[b.id]
    return axis @ (b.xpos - a.xpos) - radius_a - radius_b


def camera(name='overview'):
    c=mujoco.MjvCamera(); c.type=mujoco.mjtCamera.mjCAMERA_FREE
    if name=='slot':
        c.lookat[:]=[.370,.048,1.080]; c.distance=.098; c.azimuth=34; c.elevation=-10
    elif name=='detail':
        c.lookat[:]=[.32,.015,1.015]; c.distance=.53; c.azimuth=165; c.elevation=-21
    elif name=='hands':
        c.lookat[:]=[.24,0,.99]; c.distance=.85; c.azimuth=165; c.elevation=-21
    elif name=='top':
        c.lookat[:]=[.36,0,.97]; c.distance=.8; c.azimuth=90; c.elevation=-75
    else:
        c.lookat[:]=[.26,0,.91]; c.distance=1.76; c.azimuth=144; c.elevation=-17
    return c

def add_loop(scene,env):
    """Compatibility hook: the physical ribbon is now in the MJCF itself."""
    pass

def render(env,view='overview',width=1280,height=800):
    if env.renderer is None:
        env.renderer=mujoco.Renderer(env.model,height=height,width=width)
    opt=mujoco.MjvOption(); opt.geomgroup[3]=0; opt.sitegroup[:]=0
    env.renderer.update_scene(env.data,camera=camera(view),scene_option=opt)
    add_loop(env.renderer.scene,env)
    return env.renderer.render().copy()

def wrist_frame_size(width, height):
    scale = max(1, min(width // 4, height // 3))
    return 4 * scale, 3 * scale


def render_sensor(env,name,width=410,height=214):
    if not getattr(env,'sensor_renderer',None):
        env.sensor_renderer=mujoco.Renderer(env.model,height=height,width=width)
    opt=mujoco.MjvOption();opt.geomgroup[3]=0;opt.sitegroup[:]=0
    env.sensor_renderer.update_scene(env.data,camera=name,scene_option=opt)
    return env.sensor_renderer.render().copy()

if __name__=='__main__':
    env=G1ZipTieEnv()
    print('IK',env.plan_errors)
    while env.data.time<DURATION:
        env.step(dt=.02)
    print(json.dumps(env.info(),indent=2))

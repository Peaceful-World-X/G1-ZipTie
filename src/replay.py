"""Replay verified JOINT COMMANDS through live MuJoCo physics (not pose playback)."""
import hashlib
import numpy as np
import mujoco
from task_sequence import G1ZipTieEnv as PlannerEnv, DURATION, PHASES, SCRIPT_RATE
from simulation import ROOT

class G1ZipTieEnv(PlannerEnv):
    def __init__(self):
        path=ROOT/'recordings/actions.npz'
        if not path.exists(): raise RuntimeError('Run python src/task_sequence.py first to generate verified commands.')
        self.cache=np.load(path)
        scene=ROOT/'scenes/workcell_sequence.xml'
        if str(self.cache['scene_hash'])!=hashlib.sha256(scene.read_bytes()).hexdigest():
            raise RuntimeError('Scene changed. Regenerate the verified command recording with src/task_sequence.py.')
        if 'layout_hash' not in self.cache or str(self.cache['layout_hash'])!=hashlib.sha256((ROOT/'layout.json').read_bytes()).hexdigest():
            raise RuntimeError('Layout changed. Regenerate the verified commands with src/task_sequence.py.')
        super().__init__()
        self.initial_q=self.data.qpos.copy();self.initial_ctrl=self.data.ctrl.copy()
        self.reset()

    def reset(self,seed=42):
        if not hasattr(self,'initial_q'):return super().reset(seed)
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:]=self.initial_q;self.data.ctrl[:]=self.initial_ctrl
        self.command=self.initial_ctrl.copy();self.desired=self.command.copy()
        self.model.site_quat[:]=self.cache['sitequat'][0]
        self.model.eq_data[:]=self.cache['eqdata'][0]
        self.success=False;self.ever_entered=False;self.hold=0.;self.max_tail_contact=0.;self.max_strip_gap=0.;self.pulse_until=-1
        self.inserted=False;self.placed=False;self.failure=None;self.grasped={'left':False,'right':False};self.index=0
        self.collision_peaks={}
        mujoco.mj_forward(self.model,self.data)
        return self.observe(),self.info()

    def step(self,action=None,dt=.02):
        for _ in range(max(1,round(dt/self.model.opt.timestep))):
            while self.index<len(self.cache['times']) and self.data.time>=self.cache['times'][self.index]-1e-10:
                i=self.index; self.desired=self.cache['ctrl'][i]
                active=self.cache['eqactive'][i].copy()
                for side,obj in [('left','head'),('right','tail')]:
                    eid=self.model.equality(side+'_grasp').id
                    if active[eid] and not self.data.eq_active[eid]:
                        distance=np.linalg.norm(self.data.site(side+'_grip').xpos-self.data.site(obj+'_grasp').xpos)
                        if distance>.006:
                            active[eid]=0;self.failure=f'{side} grasp missed during live replay'
                    self.grasped[side]=bool(active[eid])
                latch=self.model.equality('thread_latch').id
                if active[latch] and not self.data.eq_active[latch] and not self.success:
                    active[latch]=0;self.failure='Live replay insertion not verified'
                self.model.site_quat[:]=self.cache['sitequat'][i]
                self.model.eq_data[:]=self.cache['eqdata'][i]
                self.data.eq_active[:]=active
                if active[latch]:self.inserted=True
                if self.inserted and self.data.time>=21.8 and not self.grasped['left']:self.placed=True
                self.index+=1
            rate=2.8*self.model.opt.timestep
            self.command+=np.clip(self.desired-self.command,-rate,rate)
            self.data.ctrl[:]=self.command
            self.data.qfrc_applied[:]=0;self.data.qfrc_applied[self.ctrl_q]=self.data.qfrc_bias[self.ctrl_q]
            self.data.xfrc_applied[:]=0
            if self.data.time<self.pulse_until:self.data.xfrc_applied[self.model.body('strip_16').id,0]=.003
            t=self.data.time
            mujoco.mj_step(self.model,self.data);mujoco.mj_forward(self.model,self.data)
            if self.data.time<t:raise RuntimeError('Numerical instability in replay')
            self.update_success()
        return self.observe(),float(self.inserted and self.placed),self.placed,self.data.time>=DURATION,self.info()

if __name__=='__main__':
    import time,json
    start=time.perf_counter();e=G1ZipTieEnv()
    while e.data.time<DURATION:e.step(dt=.02)
    print(json.dumps(e.info(),indent=2));print('Wall seconds:',time.perf_counter()-start)

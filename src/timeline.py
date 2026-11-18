"""Seek checkpoints of a verified physics rollout, including constraint/controller state."""
import hashlib
import json
import numpy as np
import mujoco
from simulation import ROOT
from replay import DURATION

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
CACHE_STEP = .01
FIELDS = ('success', 'ever_entered', 'hold', 'max_tail_contact', 'max_strip_gap',
          'pulse_until', 'inserted', 'placed', 'failure', 'grasped', 'index', 'collision_peaks')


def fingerprint():
    digest = hashlib.sha256(b'G1-timeline-v2-dense-physics')
    for name in ('scenes/workcell_sequence.xml', 'layout.json', 'recordings/actions.npz',
                 'src/replay.py', 'src/task_sequence.py', 'src/vertical_handoff.py', 'src/reference_motion.py',
                 'references/recorded_joint_states.npz', 'src/simulation.py', 'src/timeline.py'):
        digest.update((ROOT/name).read_bytes())
    digest.update(mujoco.__version__.encode())
    return digest.hexdigest()


class Timeline:
    def __init__(self, env, progress=None):
        self.env = env
        self.path = ROOT/'recordings/playback_states.npz'
        self.identity = fingerprint()
        self.frames = None
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as cached:
                if str(cached['fingerprint']) == self.identity:
                    self.frames = {key: cached[key] for key in cached.files}
        if self.frames is None:
            self.build(progress)

    def capture(self):
        e = self.env
        state = np.empty(mujoco.mj_stateSize(e.model, STATE_SPEC))
        mujoco.mj_getState(e.model, e.data, state, STATE_SPEC)
        meta = {name: getattr(e, name) for name in FIELDS}
        return dict(times=float(e.data.time), states=state,
                    sitequat=e.model.site_quat.copy(), eqdata=e.model.eq_data.copy(),
                    command=e.command.copy(), desired=e.desired.copy(),
                    metadata=json.dumps(meta), warnings=e.data.warning.number.copy(),
                    warning_info=e.data.warning.lastinfo.copy())

    def build(self, progress):
        e = self.env
        e.reset()
        rows = [self.capture()]
        for i in range(1, int(round(DURATION/CACHE_STEP))+1):
            e.step(dt=CACHE_STEP)
            rows.append(self.capture())
            if progress:
                progress(i*CACHE_STEP/DURATION)
        report = e.info()
        if not report['task_complete'] or int(e.data.warning.number.sum()):
            raise RuntimeError('Playback verification failed: '+json.dumps(report))
        self.frames = {key: np.array([row[key] for row in rows]) for key in rows[0]}
        self.frames['fingerprint'] = np.array(self.identity)
        self.frames['report'] = np.array(json.dumps(report))
        # Atomic replacement prevents partially written checkpoint files on interruption.
        temp = self.path.with_suffix('.tmp.npz')
        np.savez_compressed(temp, **self.frames)
        temp.replace(self.path)
        self.seek(0)

    def _restore(self, index):
        e=self.env
        e.model.site_quat[:] = self.frames['sitequat'][index]
        e.model.eq_data[:] = self.frames['eqdata'][index]
        mujoco.mj_setState(e.model, e.data, self.frames['states'][index], STATE_SPEC)
        e.command = self.frames['command'][index].copy()
        e.desired = self.frames['desired'][index].copy()
        for name, value in json.loads(str(self.frames['metadata'][index])).items():
            setattr(e, name, value)
        e.data.warning.number[:] = self.frames['warnings'][index]
        e.data.warning.lastinfo[:] = self.frames['warning_info'][index]

    def display(self, seconds):
        """Smooth presentation of verified physics, without rerunning its solver.

        This is explicitly cached playback. Live intervention must call seek()
        first to restore an exact integration state before stepping physics.
        """
        if not np.isfinite(seconds):raise ValueError('Display time must be finite')
        target=float(np.clip(seconds,0,DURATION));e=self.env
        times=self.frames['times']
        index=int(np.clip(np.searchsorted(times,target+1e-8,side='right')-1,0,len(times)-1))
        self._restore(index)
        if index+1<len(times) and target>times[index]+1e-8:
            # State ordering starts with time then qpos. Use MuJoCo quaternion
            # interpolation, not component-wise interpolation of free joints.
            next_q=self.frames['states'][index+1,1:1+e.model.nq]
            delta=np.empty(e.model.nv)
            mujoco.mj_differentiatePos(e.model,delta,times[index+1]-times[index],e.data.qpos,next_q)
            mujoco.mj_integratePos(e.model,e.data.qpos,delta,target-times[index])
        e.data.time=target
        mujoco.mj_forward(e.model,e.data)
        return target

    def seek(self, seconds, preview=False):
        """Restore a baseline checkpoint; explicit seeking discards manual perturbations."""
        e = self.env
        if not np.isfinite(seconds):
            raise ValueError('Seek time must be finite')
        target = float(np.clip(seconds, 0, DURATION))
        times = self.frames['times']
        index = int(np.clip(np.searchsorted(times, target+1e-8, side='right')-1, 0, len(times)-1))
        mujoco.mj_resetData(e.model, e.data)
        self._restore(index)
        mujoco.mj_forward(e.model, e.data)
        if not preview and target-e.data.time > e.model.opt.timestep/2:
            e.step(dt=target-e.data.time)
        return e.data.time


if __name__ == '__main__':
    from replay import G1ZipTieEnv
    e = G1ZipTieEnv()
    last = [-1]
    def progress(fraction):
        pct = int(fraction*100)//10*10
        if pct != last[0]:
            print(f'Preparing seek checkpoints: {pct}%', flush=True)
            last[0] = pct
    try:
        timeline = Timeline(e, progress)
        print(timeline.path, flush=True)
        print(str(timeline.frames['report']), flush=True)
    finally:
        e.close()

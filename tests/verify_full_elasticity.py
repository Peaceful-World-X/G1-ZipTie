"""Check elasticity in the final full-task scene with the grasped loop held still."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'src'))
from replay import G1ZipTieEnv,SCRIPT_RATE
from simulation import ROOT
import mujoco
import numpy as np
e=G1ZipTieEnv()
from timeline import Timeline
Timeline(e).seek(18.5)
assert e.inserted and not e.failure
def advance(duration,force=0):
    samples=[]
    for _ in range(round(duration/e.model.opt.timestep)):
        e.data.qfrc_applied[:]=0
        e.data.qfrc_applied[e.ctrl_q]=e.data.qfrc_bias[e.ctrl_q]
        e.data.xfrc_applied[:]=0
        e.data.xfrc_applied[e.model.body('strip_16').id,0]=force
        mujoco.mj_step(e.model,e.data);mujoco.mj_forward(e.model,e.data)
        samples.append(e.data.site_xpos[e.strip_sites].copy())
    return np.array(samples)
advance(1)
baseline=e.data.site_xpos[e.strip_sites].copy()
response=np.concatenate([advance(.12,.003),advance(3)])
displacements=np.linalg.norm(response-baseline,axis=2).max(axis=1)
report={'force_N':.003,'pulse_seconds':.12,'peak_displacement_mm':float(displacements.max()*1000),'residual_after_3s_mm':float(displacements[-1]*1000),'warnings':int(e.data.warning.number.sum())}
assert report['peak_displacement_mm']>.1,report
assert report['residual_after_3s_mm']<report['peak_displacement_mm']*.6,report
assert not report['warnings'],report
(ROOT/'recordings/elasticity_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2));e.close()

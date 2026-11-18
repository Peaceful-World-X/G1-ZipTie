"""Find reachable head presentation poses before committing a full rollout."""
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from task_sequence import G1ZipTieEnv,ROOT

e=G1ZipTieEnv();f=np.load(ROOT/'recordings/planner_failure_state.npz')
rel=R.from_quat(f['sitequat'][e.model.site('head_grasp').id][[1,2,3,0]]).as_matrix().T
hgoal=e.reference.pose(12,'left')[1]@rel
tailgoal=hgoal@R.from_euler('y',-90,degrees=True).as_matrix()@R.from_euler('z',180,degrees=True).as_matrix()
data=mujoco.MjData(e.model);results=[]
def solve(side,pos,rot):
    ids=e.arms[side];site=e.model.site(side+'_grip').id
    ref=e.reference.joints(8,side);lo=e.model.jnt_range[ids,0]+.01;hi=e.model.jnt_range[ids,1]-.01
    def fun(q):
        data.qpos[ids]=q;mujoco.mj_kinematics(e.model,data)
        return np.r_[data.site_xpos[site]-pos,.25*R.from_matrix(rot@data.site_xmat[site].reshape(3,3).T).as_rotvec()]
    result=None
    for stamp in (8,2,5,12):
        candidate=least_squares(fun,np.clip(e.reference.joints(stamp,side),lo,hi),bounds=(lo,hi),max_nfev=100,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        if result is None or np.linalg.norm(candidate.fun)<np.linalg.norm(result.fun):result=candidate
        if np.linalg.norm(result.fun)<.0001:break
    error=np.linalg.norm(fun(result.x));score=np.linalg.norm(result.x-ref)
    return error,score,result.x.tolist()
for yaw in (-120,-90,-60):
 for pitch in (45,65,85):
  for roll in (-150,-90,-30,30,90,150):
   H=R.from_euler('xyz',[roll,pitch,yaw],degrees=True).as_matrix();p=np.array([.28,.09,1.08])
   lp=p+H@np.array([.008,0,-.004]);lr=H@rel.T
   rp=p+H@np.array([.176,0,-.004]);rr=H@tailgoal.T@e.reference.pose(12,'right')[1]
   left=solve('left',lp,lr)
   right=solve('right',rp,rr)
   results.append(dict(angles=[roll,pitch,yaw],position=p.tolist(),score=left[1]+right[1]+1000*(left[0]+right[0]),errors_mm=[left[0]*1000,right[0]*1000],left=left[2],right=right[2]))
results.sort(key=lambda x:x['score'])
(ROOT/'recordings/regrasp_candidates.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results[:5],indent=2));e.close()

"""Static collision/reachability screen for the upright left grasp."""
import numpy as np,mujoco,json
from scipy.spatial.transform import Rotation as R
from task_sequence import G1ZipTieEnv,ROOT
from vertical_handoff import tail_pose
e=G1ZipTieEnv();f=np.load(ROOT/'recordings/planner_failure_state.npz');e.data.qpos[:]=f['qpos'];mujoco.mj_forward(e.model,e.data)
head=e.data.body('lock_head');hp=head.xpos.copy();H=head.xmat.reshape(3,3).copy()
tp=e.data.site('tail_grasp').xpos.copy();B=e.data.body('tail_body').xmat.reshape(3,3).copy();lp=e.data.site('head_grasp').xpos.copy()
e.model.jnt_range[e.arms['right'][1],1]=.18
rows=[]
for dx in (.02,.04):
 for dz in (.02,.04,.06):
  shift=np.array([dx,0,dz])
  for yaw in (-65,-50,-35):
   L=R.from_euler('z',yaw,degrees=True).as_matrix()@R.from_euler('y',-15,degrees=True).as_matrix()@R.from_euler('x',-90,degrees=True).as_matrix()
   base=e.data.qpos.copy();base[e.arms['left']]=e.reference.joints(5,'left');base,le=e.solve_ik('left',lp+shift,L,base)
   e.ikdata.qpos[:]=base;mujoco.mj_forward(e.model,e.ikdata)
   v=e.ikdata.body('left_wrist_yaw_link').xpos-e.ikdata.body('left_elbow_link').xpos
   tilt=float(np.degrees(np.arctan2(abs(v[2]),np.linalg.norm(v[:2]))))
   if le>.001 or tilt>27:continue
   for theta in (60,70,80):
    a=np.radians(theta);P=np.array([[0,0,1],[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0]])
    measurements=[];qs=[]
    for name,pos,rot in [('grasp',tp+shift,B@P),('pre',tail_pose((hp+shift,H))[0],tail_pose((hp+shift,H))[1]@P),('insert',tail_pose((hp+shift,H),.031)[0],tail_pose((hp+shift,H),.031)[1]@P)]:
     q=base.copy();q[e.arms['right']]=e.reference.joints(12,'right');q,err=e.solve_ik('right',pos,rot,q);e.ikdata.qpos[:]=q;mujoco.mj_forward(e.model,e.ikdata)
     contacts=[-c.dist*1000 for c in e.ikdata.contact if c.dist<0 and ((e.model.geom_contype[c.geom1]|e.model.geom_contype[c.geom2])&16)]
     measurements.append([name,err*1000,max(contacts,default=0)]);qs.append(q[e.arms['right']].tolist())
    score=max(x[1] for x in measurements)+max(x[2] for x in measurements)
    rows.append(dict(dx=dx,dz=dz,left_yaw=yaw,theta=theta,left_tilt=tilt,score=score,checks=measurements,qright=qs))
rows.sort(key=lambda x:x['score'])
(ROOT/'recordings/vertical_workspace_candidates.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print(json.dumps(rows[:4],indent=2));e.close()

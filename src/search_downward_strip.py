"""Check natural left-forearm poses with a straight, downward-held ribbon."""
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from task_sequence import G1ZipTieEnv, ROOT
from vertical_handoff import tail_pose,left_rotation
def tail_pose(head,distance=.067):
 B=head[1]@R.from_euler('y',90,degrees=True).as_matrix()
 return head[0]+head[1][:,2]*distance,B

e=G1ZipTieEnv()
# This kinematic search uses the original handoff pose at 5.5 s.
with np.load(ROOT/'references/reachability_seeds.npz') as seeds:
    e.data.qpos[:]=seeds['downward_handoff_qpos']
mujoco.mj_forward(e.model,e.data)
H0=e.data.body('lock_head').xmat.reshape(3,3).copy()
p0=e.data.site('head_grasp').xpos.copy()
G0=left_rotation();relative_left=G0.T@H0
e.model.jnt_range[e.model.joint('right_shoulder_roll_joint').id,1]=.18
def metrics(q):
 e.ikdata.qpos[:]=q;mujoco.mj_forward(e.model,e.ikdata)
 v=e.ikdata.body('left_wrist_yaw_link').xpos-e.ikdata.body('left_elbow_link').xpos
 tilt=float(np.degrees(np.arctan2(abs(v[2]),np.linalg.norm(v[:2]))))
 contacts=[-c.dist*1000 for c in e.ikdata.contact if c.dist<0 and ((e.model.geom_contype[c.geom1]|e.model.geom_contype[c.geom2])&16)]
 return tilt,max(contacts,default=0)
rows=[];left_rows=[]
for hyaw in (-90,):
 H=R.from_euler('z',hyaw,degrees=True).as_matrix()@R.from_euler('y',90,degrees=True).as_matrix()
 for gyaw in (-65,-50,-35):
  for groll in (90,):
   hx=.37;hz=1.08;hp=np.array([hx,.05,hz])
   G=R.from_euler('z',gyaw,degrees=True).as_matrix()@R.from_euler('y',-15,degrees=True).as_matrix()@R.from_euler('x',groll,degrees=True).as_matrix()
   G0=H0@H.T@G
   base=e.data.qpos.copy();base[e.arms['left']]=e.reference.joints(5,'left')
   before,err0=e.solve_ik('left',p0,G0,base,iterations=100)
   after,err1=e.solve_ik('left',hp+H@e.model.site('head_grasp').pos,G,before,iterations=100)
   tilt,contact=metrics(after)
   left_rows.append(dict(head_yaw=hyaw,gyaw=gyaw,groll=groll,hx=hx,hz=hz,err0=err0,err1=err1,tilt=tilt,contact=contact))
   if max(err0,err1)>.001 or tilt>25 or contact>.5:continue
   for theta in (-80,-50,-20,10):
    ang=np.radians(theta);P=np.array([[0,0,1],[np.cos(ang),-np.sin(ang),0],[np.sin(ang),np.cos(ang),0]])
    checks=[]
    relative=np.array(e.layout['strip_initial_endpoint'])+e.model.site('tail_grasp').pos-e.model.site('tail_root').pos
    for name,p,B in [('grasp',hp+H@relative,H),('pre',*tail_pose((hp,H))),('insert',*tail_pose((hp,H),.031))]:
     q=after.copy();q[e.arms['right']]=e.reference.joints(12,'right')
     # tail_grasp is 8 mm behind its body's origin.
     if name!='grasp':p=p+B@e.model.site('tail_grasp').pos
     q,err=e.solve_ik('right',p,B@P,q,iterations=100)
     _,c=metrics(q);checks.append([name,err*1000,c])
    score=max(x[1] for x in checks)+max(x[2] for x in checks)
    rows.append(dict(head_yaw=hyaw,gyaw=gyaw,groll=groll,hx=hx,hz=hz,theta=theta,tilt=tilt,score=score,checks=checks,left_before=before[e.arms['left']].tolist(),left_after=after[e.arms['left']].tolist()))
rows.sort(key=lambda x:x['score'])
(ROOT/'recordings/downward_candidates.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print(json.dumps(rows[:6],indent=2));e.close()
(ROOT/'recordings/downward_left_candidates.json').write_text(json.dumps(left_rows,indent=2),encoding='utf-8')

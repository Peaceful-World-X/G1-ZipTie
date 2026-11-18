"""Reachability screen for horizontal jaws and a 180-degree wrist flip."""
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from task_sequence import G1ZipTieEnv,ROOT

e=G1ZipTieEnv()
with np.load(ROOT/'references/reachability_seeds.npz') as seeds:
    e.data.qpos[:]=seeds['d405_handoff_qpos']
mujoco.mj_forward(e.model,e.data)
hp0=e.data.body('lock_head').xpos.copy();H0=e.data.body('lock_head').xmat.reshape(3,3).copy()
hp=np.array([.37,.05,1.08]);H=R.from_euler('zy',[-90,90],degrees=True).as_matrix()
# Explicit matrix order: yaw in world, then pitch in the local frame.
H=R.from_euler('z',-90,degrees=True).as_matrix()@R.from_euler('y',90,degrees=True).as_matrix()
def metrics(q):
 e.ikdata.qpos[:]=q;mujoco.mj_forward(e.model,e.ikdata)
 v=e.ikdata.body('left_wrist_yaw_link').xpos-e.ikdata.body('left_elbow_link').xpos
 tilt=float(np.degrees(np.arctan2(abs(v[2]),np.linalg.norm(v[:2]))))
 cs=[]
 for c in e.ikdata.contact:
  if c.dist<0 and ((e.model.geom_contype[c.geom1]|e.model.geom_contype[c.geom2])&16):
   cs.append([e.model.geom(c.geom1).name,e.model.geom(c.geom2).name,-c.dist*1000])
 return tilt,cs
rows=[]
for yaw in (-90,-75,-60):
 for tilt in (0,15):
  G=R.from_euler('z',yaw,degrees=True).as_matrix()@R.from_euler('y',-tilt,degrees=True).as_matrix()
  base=e.data.qpos.copy();base[e.arms['left']]=e.reference.joints(5,'left')
  post,err=e.solve_ik('left',hp+H@e.model.site('head_grasp').pos,G,base)
  ft,cs=metrics(post)
  pres=[]
  for dz in (0,.05,.1):
   pre,ep=e.solve_ik('left',hp0+H0@e.model.site('head_grasp').pos+[0,0,dz],H0@H.T@G,post)
   pt,pc=metrics(pre);pres.append(dict(dz=dz,error=ep,tilt=pt,contacts=pc))
  rows.append(dict(yaw=yaw,tilt=tilt,error=err,forearm=ft,contacts=cs,pre=pres,q=post[e.arms['left']].tolist()))
(ROOT/'recordings/d405_posture_search.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print(json.dumps(rows,indent=2));e.close()

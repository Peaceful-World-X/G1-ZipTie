import numpy as np
from simulation import *
from PIL import Image
env=G1ZipTieEnv()
print('initial',env.info(),flush=True)
for side in ('left','right'):
 print(side,env.data.qpos[env.arms[side]],flush=True)
(ROOT/'recordings').mkdir(exist_ok=True)
Image.fromarray(render(env)).save(ROOT/'recordings/debug_start.png')
while env.data.time<18:
 env.step(dt=.02)
print('final',env.info(),flush=True)
print('qerror',env.data.qpos-env.plan[-1][1],flush=True)
print('contacts',[(env.model.geom(c.geom1).name,env.model.geom(c.geom2).name,c.dist) for c in env.data.contact],flush=True)
Image.fromarray(render(env)).save(ROOT/'recordings/debug_final.png')
env.close()

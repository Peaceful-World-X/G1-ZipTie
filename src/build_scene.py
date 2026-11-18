"""Build our workcell from the unmodified, licensed Menagerie G1 model."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]

def element(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})

def vec(v):
    return ' '.join(f'{x:.10g}' for x in v)

def add_dex1(root,tool,side):
    """Import official URDF geometry, inertias, axes and joint limits unchanged."""
    directory=ROOT/'assets/vendor/unitree_ros/robots/dexterous_hand_description/dex1_1'
    urdf=ET.parse(directory/'dex1_1.urdf').getroot()
    asset=root.find('asset')
    for name in ('Dex1_base_link','Dex1_finger_link_1','Dex1_finger_link_2'):
        if asset.find(f'mesh[@name="{name}"]') is None:
            element(asset,'mesh',name=name,file=str((directory/'meshes'/f'{name}.STL').resolve()))
    base=element(tool,'body',name=f'{side}_dex1',pos='.0415 0 0')
    for i,name in enumerate(('Dex1_base_link','Dex1_finger_link_1','Dex1_finger_link_2')):
        link=urdf.find(f'link[@name="{name}"]')
        body=base
        if i:
            joint=urdf.find(f'joint[@name="Dex1_finger_joint_{i}"]')
            body=element(base,'body',name=f'{side}_dex1_finger_{i}',pos=joint.find('origin').get('xyz'))
            jname=f'{side}_jaw_{i}'
            element(body,'joint',name=jname,type='slide',axis=joint.find('axis').get('xyz'),range='0 .041',damping='2',frictionloss='.01',armature='.002')
            element(root.find('actuator'),'position',name=jname,joint=jname,kp='250',kv='4',ctrlrange='0 .041',forcerange='-20 20')
        inertia=link.find('inertial'); tensor=inertia.find('inertia')
        element(body,'inertial',pos=inertia.find('origin').get('xyz'),mass=inertia.find('mass').get('value'),fullinertia=' '.join(tensor.get(n) for n in ('ixx','iyy','izz','ixy','ixz','iyz')))
        element(body,'geom',type='mesh',mesh=name,rgba='.22 .25 .29 1' if not i else '.065 .075 .09 1',contype='0',conaffinity='0',group='2',mass='0')
    eq=root.find('equality')
    if eq is None: eq=element(root,'equality')
    element(eq,'joint',joint1=f'{side}_jaw_1',joint2=f'{side}_jaw_2',polycoef='0 1 0 0 0')

def add_elastic_strip(root,head,end_coords=None,bend_axis=2,length=.220,start_coords=None):
    """32 inertial ribbon links, anisotropic elastic hinges, connected to tail."""
    n=32
    start=np.array(start_coords if start_coords is not None else [-.004,0,-.004]); end=np.array(end_coords if end_coords is not None else [.189,-.030,-.030])
    ts=np.linspace(0,1,2001)
    def curve(h):
        p=start[None,:]+ts[:,None]*(end-start)[None,:]
        p[:,bend_axis]-=h*np.sin(np.pi*ts)
        return p
    def arclength(p): return np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    h=0. if abs(arclength(curve(0))[-1]-length)<1e-8 else brentq(lambda h: arclength(curve(h))[-1]-length,0,.15)
    pts=curve(h); s=arclength(pts)
    knots=np.column_stack([np.interp(np.linspace(0,length,n+1),s,pts[:,j]) for j in range(3)])
    parent=head; previous_R=np.eye(3); previous_p=np.zeros(3)
    for i,(p,q) in enumerate(zip(knots,knots[1:])):
        x=(q-p)/np.linalg.norm(q-p); y=np.cross([0,0,1],x); y/=np.linalg.norm(y); z=np.cross(x,y)
        R=np.column_stack([x,y,z]); rel=previous_R.T@R
        quat=Rotation.from_matrix(rel).as_quat()[[3,0,1,2]]
        body=element(parent,'body',name=f'strip_{i:02}',pos=vec(previous_R.T@(p-previous_p)),quat=vec(quat))
        # Pre-curved neutral ribbon; bending/twist stiffness are tunable approximations.
        for axis,k,axname in [('1 0 0',.0006,'twist'),('0 1 0',.004,'bend'),('0 0 1',.025,'side')]:
            element(body,'joint',name=f'strip_{i:02}_{axname}',type='hinge',axis=axis,limited='false',stiffness=str(k),springref='0',damping='.00006',armature='.0000002',frictionloss='0')
        seg=np.linalg.norm(q-p)
        element(body,'geom',name=f'strip_geom_{i:02}',type='box',pos=f'{seg/2} 0 0',size=f'{seg/2+.00004} .0015 .0004',rgba='1 .79 .06 1',density='1150',contype='8',conaffinity='1',group='0',solref='.004 1')
        element(body,'site',name=f'strip_site_{i:02}',pos=f'{seg/2} 0 0',size='.0005',rgba='0 0 0 0')
        parent=body; previous_R=R; previous_p=p
    element(parent,'site',name='strip_end',pos=f'{seg} 0 0',size='.0005',rgba='0 0 0 0')

def build():
    out = ROOT / 'scenes'
    out.mkdir(exist_ok=True)
    tex = ROOT / 'assets' / 'workcell'
    tex.mkdir(parents=True, exist_ok=True)
    im = Image.new('RGB', (1024, 1024), '#166762')
    d = ImageDraw.Draw(im)
    for i in range(0, 1024, 32):
        color = '#71aaa0' if i % 128 == 0 else '#3b8780'
        d.line((i,0,i,1024), fill=color, width=2 if i%128 == 0 else 1)
        d.line((0,i,1024,i), fill=color, width=2 if i%128 == 0 else 1)
    d.rectangle((18,18,1006,1006), outline='#b1c9b2', width=3)
    im.save(tex / 'cutting_mat.png')
    source = ROOT / 'assets/vendor/mujoco_menagerie/unitree_g1/g1.xml'
    tree = ET.parse(source)
    root = tree.getroot()
    root.set('model', 'G1 | Zip-tie workcell')
    root.find('compiler').set('meshdir', '../assets/vendor/mujoco_menagerie/unitree_g1/assets')
    root.find('option').set('timestep', '0.001')
    root.find('option').set('iterations', '80')
    root.find('option').set('gravity', '0 0 -9.81')
    root.find('.//default[@class="g1"]/joint').set('frictionloss','.02')
    root.find('.//default[@class="g1"]/position').set('kp','900')
    if root.find('keyframe') is not None:
        root.remove(root.find('keyframe'))
    pelvis = root.find('.//body[@name="pelvis"]')
    pelvis.remove(pelvis.find('freejoint'))
    # Demo fixture: fixed pelvis, no unvalidated humanoid self-collision control.
    for g in root.findall('.//geom[@class="collision"]'):
        g.set('contype', '0'); g.set('conaffinity', '0')
    for g in root.findall('.//geom[@class="foot"]'):
        g.set('contype', '0'); g.set('conaffinity', '0')
    for side in ('left','right'):
        wrist = root.find(f'.//body[@name="{side}_wrist_yaw_link"]')
        for g in list(wrist.findall('geom')):
            if 'rubber_hand' in g.get('mesh',''):
                wrist.remove(g)
        tool = element(wrist,'body',name=f'{side}_tool')
        add_dex1(root,tool,side)
        element(tool,'site',name=f'{side}_grip',pos='.181 0 0',size='.001',rgba='0 0 0 0')
        if side == 'left':
            head=element(tool,'body',name='lock_head',pos='.187 0 0')
            # Four boxes form a real open collision channel along local x.
            for name,pos,size in [('top','0 0 .0026','.004 .005 .0014'),('bottom','0 0 -.0026','.004 .005 .0014'),('a','0 .00375 0','.004 .00125 .0012'),('b','0 -.00375 0','.004 .00125 .0012')]:
                element(head,'geom',name='lock_'+name,type='box',pos=pos,size=size,rgba='1 .73 .06 1',mass='.00008',contype='2',conaffinity='4',friction='.45 .01 .001',solref='.004 1')
            element(head,'site',name='hole',size='.002',rgba='0 0 0 0')
            element(head,'site',name='loop_anchor',pos='-.004 0 -.004',size='.001',rgba='0 0 0 0')
            add_elastic_strip(root,head)
        else:
            element(tool,'geom',name='rigid_tail',type='box',pos='.203 0 0',size='.022 .0015 .0004',rgba='1 .79 .08 1',mass='.00012',contype='4',conaffinity='2',friction='.4 .01 .001',solref='.004 1')
            element(tool,'site',name='tip',pos='.225 0 0',size='.001',rgba='0 0 0 0')
            element(tool,'site',name='tail_root',pos='.181 0 0',size='.001',rgba='0 0 0 0')
    equality=root.find('equality')
    element(equality,'connect',name='strip_grasp',site1='strip_end',site2='tail_root',solref='.003 1',solimp='.995 .999 .0001')
    asset=root.find('asset')
    element(asset,'texture',name='sky',type='skybox',builtin='gradient',rgb1='.16 .21 .29',rgb2='.035 .05 .075',width='512',height='3072')
    element(asset,'texture',name='mat_tex',type='2d',file='../assets/workcell/cutting_mat.png')
    element(asset,'material',name='mat',texture='mat_tex',texrepeat='1 1',specular='.15',shininess='.1')
    visual=element(root,'visual')
    element(visual,'global',offwidth='1600',offheight='1000',azimuth='145',elevation='-18')
    element(visual,'quality',shadowsize='4096',offsamples='4')
    element(visual,'headlight',diffuse='.65 .65 .65',ambient='.3 .3 .3',specular='.2 .2 .2')
    element(visual,'map',znear='.01',zfar='30')
    world=root.find('worldbody')
    element(world,'light',pos='1 -2 4',dir='-.2 .5 -1',diffuse='.9 .85 .75',castshadow='true')
    element(world,'light',pos='-1 2 3',dir='.3 -.5 -1',diffuse='.4 .55 .7',castshadow='false')
    element(world,'geom',name='floor',type='plane',size='4 4 .1',rgba='.14 .18 .22 1')
    element(world,'geom',name='tabletop',type='box',pos='.53 0 .73',size='.37 .54 .025',rgba='.57 .43 .28 1')
    element(world,'geom',name='cutting_mat',type='box',pos='.51 0 .757',size='.31 .46 .002',material='mat')
    for x in (.24,.82):
        for y in (-.45,.45):
            element(world,'geom',type='box',pos=f'{x} {y} .355',size='.025 .025 .355',rgba='.12 .15 .19 1')
    # Reference-inspired cardboard sorting trays.
    for y in (-.35,.35):
        tray=element(world,'body',pos=f'.67 {y} .77')
        element(tray,'geom',type='box',size='.14 .085 .006',rgba='.5 .32 .15 1')
        for x2 in (-.137,.137):
            element(tray,'geom',type='box',pos=f'{x2} 0 .032',size='.003 .085 .032',rgba='.64 .44 .23 1')
        for y2 in (-.082,.082):
            element(tray,'geom',type='box',pos=f'0 {y2} .032',size='.14 .003 .032',rgba='.64 .44 .23 1')
    element(world,'geom',name='base_fixture',type='box',pos='-.10 0 .39',size='.035 .1 .39',rgba='.09 .13 .17 1')
    element(world,'geom',name='base_plate',type='box',pos='-.05 0 .01',size='.23 .23 .01',rgba='.1 .14 .18 1')
    # Stored cameras usable by render(), dataset clients and native viewer.
    element(world,'camera',name='overview',pos='2.05 -2.45 1.6',xyaxes='.77 .64 0 -.14 .17 .975',fovy='38')
    element(world,'camera',name='detail',pos='1.05 -.8 1.6',xyaxes='.75 .66 0 -.35 .4 .85',fovy='28')
    ET.indent(tree, space='  ')
    tree.write(out/'workcell.xml',encoding='utf-8',xml_declaration=True)
    print(out/'workcell.xml')

if __name__ == '__main__':
    build()

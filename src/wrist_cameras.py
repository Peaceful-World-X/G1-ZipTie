"""Official D405 mesh, wrist-fixed mount and an ideal rendered camera.

The URDF's nominal mesh transform is retained. Its inertia is explicitly marked
unreliable upstream, so use the product's 60 g mass and a uniform-case estimate.
"""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from build_scene import ROOT,element,vec

SOURCE='https://github.com/realsenseai/realsense-ros/tree/960306819f45006ca3f903aa046bed32c509679a/realsense2_description'

def add_wrist_d405(root,tool,side):
    config=json.loads((ROOT/'camera_mounts.json').read_text(encoding='utf-8'))
    cfg=config[side]
    center=np.array(cfg['case_center_tool_m'],float)
    target=np.array(cfg['look_at_tool_m'],float)
    optical_in_case=np.array([.0121,.009,0.])
    # Fixed clocking around the optical axis sets the fingers' orientation in the image.
    clocking = Rotation.from_euler('x', cfg['roll_deg'], degrees=True).as_matrix()
    point=center.copy()
    for _ in range(12):
        forward=target-point;forward/=np.linalg.norm(forward)
        left=np.cross([0,0,1],forward);left/=np.linalg.norm(left)
        up=np.cross(forward,left)
        rotation=np.column_stack([forward,left,up]) @ clocking
        point=center+rotation@optical_in_case
    origin=point-rotation@np.array([.0043,0,0])
    asset=root.find('asset')
    if asset.find('mesh[@name="realsense_d405"]') is None:
        element(asset,'mesh',name='realsense_d405',file=str(ROOT/'assets/vendor/realsense_ros/d405.stl'),scale='.001 .001 .001')
    body=element(tool,'body',name=f'{side}_d405',pos=vec(origin),quat=vec(Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]))
    size=np.array([.023,.042,.042]);mass=.060
    inertia=mass/12*np.array([size[1]**2+size[2]**2,size[0]**2+size[2]**2,size[0]**2+size[1]**2])
    element(body,'inertial',mass=str(mass),pos='-.0078 -.009 0',diaginertia=vec(inertia))
    meshquat=Rotation.from_euler('xyz',[90,0,90],degrees=True).as_quat()[[3,0,1,2]]
    element(body,'geom',name=side+'_d405_case',type='mesh',mesh='realsense_d405',pos='.0038 -.009 0',quat=vec(meshquat),rgba='.68 .71 .74 1',mass='0',contype='0',conaffinity='0',group='2')
    element(body,'geom',name=side+'_d405_collision',type='box',pos='-.0078 -.009 0',size='.0115 .021 .021',mass='0',contype='16',conaffinity='17',group='3')
    element(body,'geom',name=side+'_d405_face',type='box',pos='.00385 -.009 0',size='.00008 .018 .018',rgba='.025 .03 .038 1',mass='0',contype='0',conaffinity='0',group='2')
    for i,y in enumerate((0,-.018)):
        element(body,'geom',name=f'{side}_d405_lens_{i}',type='cylinder',pos=vec([.00405,y,0]),size='.0048 .00012',quat='.7071067812 0 .7071067812 0',rgba='.07 .13 .19 1',mass='0',contype='0',conaffinity='0',group='2')
    # The optical viewpoint is just in front of the cover, avoiding rendering
    # the opaque housing over its own image. No real lens/noise calibration.
    element(body,'camera',name=side+'_wrist_cam',pos='.0043 0 0',xyaxes='0 -1 0 0 0 1',fovy=str(config['vertical_fov_deg']))
    foot=np.array(cfg['mount_foot_tool_m'])
    base=center-rotation[:,2]*.022
    element(tool,'geom',name=side+'_d405_mount',type='capsule',fromto=vec(np.r_[foot,base]),size='.0045',rgba='.10 .12 .15 1',mass='.008',contype='16',conaffinity='17',group='2')
    element(tool,'geom',name=side+'_d405_mount_plate',type='box',pos=vec(foot),size='.015 .017 .002',rgba='.10 .12 .15 1',mass='.002',contype='16',conaffinity='17',group='2')
    # Black rear cover and a short strain-relieved USB lead back to the wrist.
    visual = dict(mass='0', contype='0', conaffinity='0', group='2')
    element(body,'geom',type='box',pos='-.0194 -.009 0',size='.0005 .019 .019',
            rgba='.055 .065 .075 1',**visual)
    port = origin + rotation @ np.array([-.012, .006, .018])
    end = foot + np.array([-.040, .013, -.006])
    bend = (port+end)/2 + np.array([-.025, .01, .044])
    t = np.linspace(0,1,10)[:,None]
    lead = (1-t)**2*port + 2*t*(1-t)*bend + t**2*end
    for start,finish in zip(lead[:-1],lead[1:]):
        element(tool,'geom',type='capsule',fromto=vec(np.r_[start,finish]),size='.0023',
                rgba='.018 .020 .022 1',**visual)
    return body

"""Visual checkpoints plus straightness, mirror symmetry and contact measurements."""
import json
import numpy as np
from PIL import Image,ImageDraw
from replay import G1ZipTieEnv
from timeline import Timeline
from simulation import ROOT,render,camera,projected_box_gap
import mujoco
from record_demo import font

e=G1ZipTieEnv();timeline=Timeline(e)
try:
    times=[0,2,5.5,6.2,8.4,9.2,11.5,17.8,21,24]
    sheet=Image.new('RGB',(2000,640),'#0b131d');draw=ImageDraw.Draw(sheet);rows=[]
    for i,t in enumerate(times):
        timeline.seek(t)
        rgb=render(e,'hands' if 4<=t<=20 else 'overview',width=800,height=600)
        x=(i%5)*400;y=(i//5)*320
        sheet.paste(Image.fromarray(rgb).resize((400,300)),(x,y+20))
        draw.text((x+8,y+4),f'{t:.1f}s / {e.phase()[1]}',fill='white')
        points=e.data.site_xpos[e.strip_sites];axis=points[-1]-points[0];axis/=np.linalg.norm(axis)
        delta=e.data.body('lock_head').xpos-e.data.body('tail_body').xpos
        tilt=float(np.degrees(np.arctan2(abs(delta[2]),np.linalg.norm(delta[:2]))))
        clearance=float((e.data.site('hole').xpos[2]-e.data.site('left_grip').xpos[2])*1000)
        v=e.data.body('left_wrist_yaw_link').xpos-e.data.body('left_elbow_link').xpos
        forearm_tilt=float(np.degrees(np.arctan2(abs(v[2]),np.linalg.norm(v[:2]))))
        vertical=float(abs(e.data.site('left_grip').xmat.reshape(3,3)[2,1]))
        rows.append({'time':t,'strip_line_deviation_mm':float(np.linalg.norm(np.cross(points-points[0],axis),axis=1).max()*1000),'strip_tilt_degrees':tilt,'strip_axis_from_vertical_deg':float(np.degrees(np.arccos(np.clip(-axis[2],-1,1)))),'lock_above_left_grip_mm':clearance,'left_forearm_tilt_deg':forearm_tilt,'left_jaw_vertical_alignment':vertical,'grasped':dict(e.grasped),**e.info()})
        for side in ('left','right'):
            tool=e.data.body(side+'_tool')
            case=e.data.body(side+'_d405')
            rot=tool.xmat.reshape(3,3)
            rows[-1][side+'_camera_above_wrist_mm']=float((case.xipos[2]-tool.xpos[2])*1000)
            rows[-1][side+'_wrist_roll_deg']=float(np.degrees(np.arctan2(rot[2,1],rot[2,2])))
            rows[-1][side+'_finger_elevation_deg']=float(np.degrees(np.arcsin(rot[2,0])))
    sheet.save(ROOT/'recordings/quality_contact_sheet.png')
    # Photograph both faces from the same replay used by the interactive viewer.
    pair = Image.new('RGB', (1280, 532), '#101923')
    labels = ImageDraw.Draw(pair)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    opt.sitegroup[:] = 0
    for i, (second, label) in enumerate(((13.8, '入口侧 · 带尖与孔口同高'), (17.0, '出口侧 · 带尖已穿出'))):
        timeline.display(second)
        cam = camera('slot')
        cam.lookat[:] = e.data.site('hole').xpos
        if i == 1:
            cam.azimuth, cam.elevation = -148, -8
        e.renderer.update_scene(e.data, camera=cam, scene_option=opt)
        panel = Image.fromarray(e.renderer.render()).resize((640, 480), Image.Resampling.LANCZOS)
        pair.paste(panel, (640*i, 52))
        labels.text((640*i + 16, 11), f'{second:.1f} s   {label}', font=font(22), fill='#eef2f3')
    pair.save(ROOT/'recordings/锁扣侧向穿入.png')
    timeline.seek(24)
    mirror=np.array([1,-1,-1,1,-1,1,-1])
    mirror_error=float(np.abs(e.data.qpos[e.arms['right']]-e.data.qpos[e.arms['left']]*mirror).max())
    report={'checkpoints':rows,'final_arm_mirror_error_rad':mirror_error,'collision_peaks_mm':e.collision_peaks.copy()}
    timeline.seek(8.4)
    held_position=e.data.site('left_grip').xpos.copy()
    max_drift=0.
    for second in np.arange(8.4,19.,.02):
        timeline.display(second)
        max_drift=max(max_drift,np.linalg.norm(e.data.site('left_grip').xpos-held_position))
    report['max_left_grip_drift_while_holding_mm']=float(max_drift*1000)
    tips, normals, clearances = [], [], []
    for second in np.arange(12.8, 17.81, .02):
        timeline.display(second)
        tips.append(e.data.site('tip').xpos.copy())
        normals.append(e.data.site('hole').xmat.reshape(3, 3)[:,0].copy())
        for part in ('neck_bridge', 'grasp_neck'):
            neck_axis = e.data.body('lock_head').xmat.reshape(3, 3)[:,0]
            clearances.append(projected_box_gap(e, 'rigid_tail', part, neck_axis))
    report['side_insertion'] = {
        'sample_range_s': [12.8, 17.8],
        'advance_along_hole_mm': float((tips[0] - tips[-1]) @ normals[0] * 1000),
        'tip_height_range_mm': float(np.ptp(np.asarray(tips)[:,2])*1000),
        'max_hole_axis_elevation_deg': float(np.degrees(np.arcsin(np.abs(normals).max(axis=0)[2]))),
        'min_projected_tongue_to_neck_gap_mm': float(min(clearances)*1000),
    }
    (ROOT/'recordings/quality_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'single_hand_line_deviation_mm':rows[2]['strip_line_deviation_mm'],'horizontal_handoff_tilt_deg':rows[2]['strip_tilt_degrees'],'final_arm_mirror_error_rad':mirror_error,'collision_peaks_mm':e.collision_peaks},indent=2))
finally:e.close()

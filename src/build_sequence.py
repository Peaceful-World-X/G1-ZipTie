"""Free object assembly with switchable Dex1 grasp constraints for the full task."""
import copy
import json
from build_scene import *

def build_sequence():
    build()
    layout=json.loads((ROOT/'layout.json').read_text(encoding='utf-8'))
    tree=ET.parse(ROOT/'scenes/workcell.xml'); root=tree.getroot(); world=root.find('worldbody'); eq=root.find('equality')
    left=root.find('.//body[@name="left_tool"]'); right=root.find('.//body[@name="right_tool"]')
    head=left.find('body[@name="lock_head"]'); left.remove(head)
    initial_rot=Rotation.from_euler('z',180,degrees=True)*Rotation.from_euler('y',layout.get('initial_tie_pitch_degrees',0),degrees=True)
    initial_quat=vec(initial_rot.as_quat()[[3,0,1,2]])
    head.set('pos',vec(layout['initial_head'])); head.set('quat',initial_quat); head.set('childclass','')
    del head.attrib['childclass']
    head.insert(0,ET.Element('joint',{'name':'head_free','type':'free','damping':'.001','armature':'.00002'})); world.append(head)
    head.remove(head.find('body[@name="strip_00"]'))
    # Short rigid neck provides a real gripping surface below the lock.
    # Side-facing through-hole in the broad face of the lock.
    from vertical_handoff import SLOT_FRAME
    slot_rotation = Rotation.from_matrix(SLOT_FRAME)
    slot_quat = vec(slot_rotation.as_quat()[[3,0,1,2]])
    for wall in head.findall('geom'):
        wall.set('quat',slot_quat)
        wall.set('pos',vec(slot_rotation.apply(np.fromstring(wall.get('pos'),sep=' '))))
    head.find('site[@name="hole"]').set('quat',slot_quat)
    element(head,'geom',name='grasp_neck',type='box',pos='.011 0 -.008',size='.003 .0015 .0004',rgba='1 .79 .06 1',density='1150',contype='8',conaffinity='1')
    # Connect the lower rim to the pinched neck, leaving both ends of the bore clear.
    bridge_start = np.array([.0028, 0., -.0035])
    bridge_end = np.array([.008, 0., -.008])
    span = bridge_end - bridge_start
    angle = np.arctan2(-span[2], span[0])
    bridge_quat = Rotation.from_euler('y', angle).as_quat()[[3,0,1,2]]
    element(head,'geom',name='neck_bridge',type='box',pos=vec((bridge_start + bridge_end)/2),quat=vec(bridge_quat),size=vec([np.linalg.norm(span)/2,.0015,.0004]),rgba='1 .79 .06 1',density='1150',contype='8',conaffinity='1')
    add_elastic_strip(root,head,end_coords=layout['strip_initial_endpoint'],bend_axis=1,length=layout['strip_length']-.018,start_coords=[.014,0,-.008])
    # Straight neutral strip, with enough bending stiffness to resist gravity sag.
    for joint in head.findall('.//joint'):
        if joint.get('name','').endswith(('_bend','_side')):joint.set('stiffness',str(.12*.162/(layout['strip_length']-.018)))
    for g in head.findall('geom'):
        g.set('conaffinity','5')
    element(head,'site',name='head_grasp',pos='.008 0 -.008',size='.001',rgba='0 0 0 0')
    tail=element(world,'body',name='tail_body',pos=vec(layout['initial_tail']),quat=initial_quat)
    element(tail,'joint',name='tail_free',type='free',damping='.0004',armature='.00001')
    for name in ('rigid_tail','tip','tail_root'):
        item=right.find(f'geom[@name="{name}"]') if name=='rigid_tail' else right.find(f'site[@name="{name}"]')
        right.remove(item); tail.append(item)
        p=np.fromstring(item.get('pos'),sep=' '); p[0]-=.181; item.set('pos',vec(p))
        if name=='rigid_tail':
            item.set('conaffinity','3');item.set('pos','.017 0 0');item.set('size','.027 .0015 .0004');item.set('mass',str(.00012*54/44))
        if name=='tail_root':item.set('pos','-.010 0 0')
    element(tail,'site',name='tail_grasp',pos='-.008 0 0',size='.001',rgba='0 0 0 0')
    # Continuous strip-to-tail tangent: a position-only connect creates an
    # unintended free hinge and cannot transmit the held tail's bending moment.
    seam=eq.find('*[@name="strip_grasp"]');seam.tag='weld'
    seam.set('torquescale','.10');seam.set('solref','.002 1')
    element(eq,'weld',name='right_grasp',site1='right_grip',site2='tail_grasp',active='false',solref='.002 1',solimp='.995 .999 .0001',torquescale='.08')
    element(eq,'weld',name='left_grasp',site1='left_grip',site2='head_grasp',active='false',solref='.002 1',solimp='.995 .999 .0001',torquescale='.08')
    # This latch preserves a verified threaded loop for placement; not tooth contact.
    element(eq,'weld',name='thread_latch',body1='lock_head',body2='tail_body',active='false',solref='.002 1',solimp='.995 .999 .0001',torquescale='.08')
    # Source and destination trays inside the arms' reachable workspace.
    for body in world.findall('body'):
        if body.get('pos') in ('.67 -0.35 .77','.67 0.35 .77'):
            world.remove(body)
    for label in ('source_tray','destination_tray'):
        cfg=layout[label]; x,y,z=cfg['half_size']
        tray=element(world,'body',name=label,pos=vec(cfg['center']))
        if label == 'destination_tray':
            from tray_stock import add_tray_stock
            add_tray_stock(root, tray)
        element(tray,'geom',name=label+'_floor',type='box',size=vec([x,y,.006]),rgba='.5 .32 .15 1')
        for sign in (-1,1):
            element(tray,'geom',type='box',pos=vec([sign*(x-.003),0,z]),size=vec([.003,y,z]),rgba='.64 .44 .23 1')
            element(tray,'geom',type='box',pos=vec([0,sign*(y-.003),z]),size=vec([x,.003,z]),rgba='.64 .44 .23 1')
    root.set('model','G1 + Dex1-1 | full zip-tie sequence')
    root.find('option').set('timestep','.0005')
    for joint in head.findall('.//joint'):
        if joint.get('name','').startswith('strip_'):
            joint.set('armature','.000002');joint.set('damping','.0003')
    # Restore robot/world and robot/self contacts. Separate convex Dex meshes
    # from visuals; the two interleaved fingers of one gripper are excluded.
    for geom in root.findall('.//geom[@class="collision"]'):
        geom.set('contype','16');geom.set('conaffinity','17')
    contact=root.find('contact')
    if contact is None:contact=element(root,'contact')
    for side in ('left','right'):
        for suffix in ('dex1','dex1_finger_1','dex1_finger_2'):
            body=root.find(f'.//body[@name="{side}_{suffix}"]')
            visual=body.find('geom')
            element(body,'geom',name=f'{side}_{suffix}_collision',type='mesh',mesh=visual.get('mesh'),
                    contype='16',conaffinity='17',group='3',mass='0',solref='.004 1')
        element(contact,'exclude',body1=f'{side}_dex1_finger_1',body2=f'{side}_dex1_finger_2')
        # Convex mesh hulls overlap inside connected wrist/shoulder assemblies.
        element(contact,'exclude',body1=f'{side}_wrist_roll_link',body2=f'{side}_wrist_yaw_link')
        for link in ('shoulder_roll','shoulder_yaw'):
            element(contact,'exclude',body1='torso_link',body2=f'{side}_{link}_link')
    # Real attached cameras: both gripper wrists, and front-down robot view.
    def cam(parent,name,pos,target,fovy):
        p=np.array(pos); z=p-np.array(target);z/=np.linalg.norm(z)
        x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
        element(parent,'camera',name=name,pos=vec(p),xyaxes=vec(np.r_[x,y]),fovy=str(fovy))
    from wrist_cameras import add_wrist_d405
    for side in ('left','right'):
        tool=root.find(f'.//body[@name="{side}_tool"]')
        add_wrist_d405(root,tool,side)
    torso=root.find('.//body[@name="torso_link"]')
    cam(torso,'front_down_cam',[.15,0,.38],[.40,0,-.04],75)
    from workcell_appearance import apply_appearance
    apply_appearance(root, layout, head, tail)
    ET.indent(tree,space='  '); tree.write(ROOT/'scenes/workcell_sequence.xml',encoding='utf-8',xml_declaration=True)
    # Compile once to bind the physical ribbon skin to its link transforms.
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(ROOT/'scenes/workcell_sequence.xml'))
    data=mujoco.MjData(model)
    names=('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')
    ref=np.array(layout['initial_arm_reference']);mirror=np.array([1,-1,-1,1,-1,1,-1])
    symmetric=(ref[:7]+ref[7:]*mirror)/2
    for side,angles in [('left',symmetric),('right',symmetric*mirror)]:
        for name,value in zip(names,angles):data.qpos[model.joint(f'{side}_{name}_joint').qposadr[0]]=value
    mujoco.mj_forward(model,data)
    from ribbon_skin import add_ribbon_skin
    add_ribbon_skin(root,model,data,end=layout['strip_initial_endpoint'][0])
    tree.write(ROOT/'scenes/workcell_sequence.xml',encoding='utf-8',xml_declaration=True)

if __name__=='__main__': build_sequence()

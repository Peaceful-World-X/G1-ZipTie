"""Horizontal handoff, straight downward left hold, then right-only tail flip."""
import numpy as np
from scipy.spatial.transform import Rotation
from simulation import smooth

PREINSERT_DISTANCE=.067
INSERT_TRAVEL=.036
# In the head frame the straight ribbon follows +X, its broad face lies
# in XY, and the aperture goes through its thickness along +Z.
SLOT_FRAME = Rotation.from_euler('y', -90, degrees=True).as_matrix()
TIP_IN_SLOT = Rotation.from_euler('z', 180, degrees=True).as_matrix()

def tail_pose(head,distance=PREINSERT_DISTANCE):
    position, head_rotation = head
    slot = head_rotation @ SLOT_FRAME
    return position + distance * slot[:,0], slot @ TIP_IN_SLOT

def left_rotation():
    return Rotation.from_euler('z',-75,degrees=True).as_matrix()

def downward_head():
    rotation=(Rotation.from_euler('z',-115,degrees=True).as_matrix()
              @Rotation.from_euler('y',90,degrees=True).as_matrix())
    return np.array([.37,.05,1.08]),rotation

def regrasp_rotation(tail_rotation):
    # Point across the workspace, with the camera on the downward side.
    normal = tail_rotation[:,0]
    yaw = np.radians(68.)
    forward = np.array([np.cos(yaw), np.sin(yaw), 0.])
    forward -= normal * np.dot(normal, forward)
    forward /= np.linalg.norm(forward)
    return np.column_stack((forward, np.cross(normal, forward), normal))

def targets(self):
    from task_sequence import mix_pose
    t=self.data.time;poses=dict(self.home);jaws={'left':.025,'right':.025}
    closed=lambda hand:self.saved.get(hand+'_jaw_closed',.001 if hand=='left' else .0015)
    if t<3:
        point=self.pose('tail_grasp')[0]
        rotation=Rotation.from_euler('y',self.layout['pickup_downward_tilt_degrees'],degrees=True).as_matrix()
        hover=(point+np.array([0,0,.12]),rotation)
        poses['right']=mix_pose(self.home['right'],hover,(t-.2)/1.1) if t<1.3 else mix_pose(hover,(point+np.array([0,0,.003]),rotation),(t-1.3)/.9)
        if t>=2.2:
            jaws['right']=closed('right')
            if not self.grasped['right']:self.capture('right')
        if self.grasped['right']:poses['right']=self.saved.setdefault('pickup',self.pose('right_grip'))
        return poses,jaws
    if 'pickup' not in self.saved:
        self.failure='Right tail pickup missed';return poses,jaws
    handoff=(np.array([.305,-.075,1.045]),Rotation.from_euler('z',-110,degrees=True).as_matrix())
    poses['right']=mix_pose(self.saved['pickup'],self.object_to_grip('right',handoff),(t-3)/2)
    jaws['right']=closed('right')
    if t>=4:
        point=self.pose('head_grasp')[0]
        receive_rotation=left_rotation()
        poses['left']=mix_pose(self.home['left'],(point,receive_rotation),(t-4)/1.8)
        if t>=5.8:
            jaws['left']=closed('left')
            if not self.grasped['left'] and t<6.5:self.capture('left')
        if self.grasped['left']:poses['left']=self.saved.setdefault('left_hold',self.pose('left_grip'))
    if t>=6.5:
        if not self.grasped['left'] and t<21.8:
            self.failure='Left upright grasp missed';return poses,jaws
        jaws['left']=closed('left')
        if 'right_release_pose' not in self.saved:
            self.saved['right_release_pose']=self.pose('right_grip')
            head=self.data.body('lock_head')
            self.saved['head_start']=(head.xpos.copy(),head.xmat.reshape(3,3).copy())
            self.saved['head_goal']=downward_head()
            self.saved['receive_relative']=self.offsets['left'][1].copy()
            self.release('right','right_release_for_regrasp')
        if t<8.2:
            # Model controlled rolling of the neck inside a soft pinch. Keep
            # its contact anchor coincident while gravity and the weld act.
            # Only the relative weld frame changes; the head remains dynamic.
            goal=self.saved['head_goal']
            target_relative=left_rotation().T @ goal[1]
            relative=mix_pose((goal[0],self.saved['receive_relative']),
                              (goal[0],target_relative),(t-6.5)/1.5)[1]
            anchor=self.model.site('head_grasp')
            anchor.quat[:]=Rotation.from_matrix(relative.T).as_quat()[[3,0,1,2]]
            self.offsets['left']=(-relative@anchor.pos,relative)
            poses['left']=(goal[0]+goal[1]@anchor.pos,left_rotation())
        elif 'vertical_hold_settled' not in self.saved:
            self.saved['vertical_hold_settled']=True
            self.saved['left_hold_joints']=self.data.qpos[self.arms['left']].copy()
            self.saved['left_hold']=self.pose('left_grip')
            head=self.data.body('lock_head')
            self.saved['head_goal']=(head.xpos.copy(),head.xmat.reshape(3,3).copy())
        if t<9.5:
            jaws['right']=.025
            point=self.pose('tail_grasp')[0]
            rotation=regrasp_rotation(self.data.body('tail_body').xmat.reshape(3,3))
            start=self.saved['right_release_pose']
            retreat=start[0]+np.array([-.03,-.025,-.025])
            poses['right']=mix_pose(start,(retreat,rotation),(t-6.5)/.8) if t<7.3 else mix_pose((retreat,rotation),(point,rotation),(t-7.3)/1.2)
            if t>=8.8:
                jaws['right']=closed('right')
                if not self.right_regrasped:self.capture('right')
                if self.right_regrasped:poses['right']=self.saved.setdefault('right_regrasp_pose',self.pose('right_grip'))
            return poses,jaws
    if t>=9.5:
        if not self.right_regrasped:
            self.failure='Right tail regrasp missed';return poses,jaws
        jaws={'left':closed('left'),'right':closed('right')}
        head=self.saved['head_goal']
        poses['left']=self.saved['left_hold']
        distance=PREINSERT_DISTANCE-INSERT_TRAVEL*smooth((t-12.6)/5.0)
        tail_position, tail_rotation = tail_pose(head, distance)
        anchor = self.model.site('tail_grasp')
        end_position = tail_position + tail_rotation @ anchor.pos
        end_rotation = Rotation.from_euler('ZY', [67, 12], degrees=True).as_matrix()
        start = self.saved.setdefault('flip_start_grip', self.pose('right_grip'))
        start_tail = self.saved.setdefault('flip_start_tail', self.data.body('tail_body').xmat.reshape(3,3).copy())
        fraction = (t - 9.5) / 3.1
        w = float(smooth(fraction))
        delta = Rotation.from_matrix(start[1].T @ end_rotation).as_rotvec()
        if delta[0] < 0:
            delta *= 1 - 2*np.pi/np.linalg.norm(delta)
        grip_rotation = start[1] @ Rotation.from_rotvec(w * delta).as_matrix()
        bow = np.array([.010, -.030, .015]) * np.sin(np.pi*w)
        poses['right'] = ((1-w)*start[0] + w*end_position + bow, grip_rotation)
        # Roll the tongue about the captured material point as the wrist turns.
        # The site weld applies forces to the dynamic tail at that same point.
        if self.grasped['right'] and t < 19:
            target_tail = mix_pose((start[0], start_tail), (end_position, tail_rotation), fraction)[1]
            relative = grip_rotation.T @ target_tail
            anchor.quat[:] = Rotation.from_matrix(relative.T).as_quat()[[3,0,1,2]]
            self.offsets['right'] = (-relative @ anchor.pos, relative)
        if t>=12.6 and self.success and not self.inserted:self.latch()
        if t>=19:
            if not self.inserted:
                self.failure='Insertion was not verified';return poses,jaws
            if self.grasped['right']:self.release('right','right_release_after_insertion')
            jaws['right']=.025
            place_rotation=self.reference.pose(19,'left')[1]@self.offsets['left'][1]
            place=(np.array(self.layout['place_head']),place_rotation)
            poses['left']=self.object_to_grip('left',mix_pose(head,place,(t-19)/2.8))
            if t>=21.8:
                if not self.placed:self.release('left','left_release_to_tray');self.placed=True
                jaws['left']=.025
    return poses,jaws

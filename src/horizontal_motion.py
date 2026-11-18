"""Video-observed sequence: horizontal handoff, release, regrasp, curl, insert."""
import numpy as np
from scipy.spatial.transform import Rotation
from simulation import smooth


def targets(self):
    from task_sequence import mix_pose
    t=self.data.time;poses=dict(self.home);openj={'left':.025,'right':.025}
    rz=lambda a:Rotation.from_euler('z',a,degrees=True).as_matrix()
    hole_local=Rotation.from_euler('y',90,degrees=True).as_matrix()
    display_head=rz(-90)
    if t<3:
        p,_=self.pose('tail_grasp')
        rot=Rotation.from_euler('y',self.layout['pickup_downward_tilt_degrees'],degrees=True).as_matrix()
        hover=(p+np.array([0,0,.12]),rot)
        poses['right']=mix_pose(self.home['right'],hover,(t-.2)/1.1) if t<1.3 else mix_pose(hover,(p+np.array([0,0,.003]),rot),(t-1.3)/.9)
        if t>=2.2:
            openj['right']=.0015
            if not self.grasped['right']:self.capture('right')
        if self.grasped['right']:poses['right']=self.saved.setdefault('pickup',self.pose('right_grip'))
        return poses,openj
    if 'pickup' not in self.saved:
        self.failure='Right tail pickup missed';return poses,openj
    # Rotate in the table plane: the head and tail stay approximately level.
    handoff=(np.array([.285,-.075,.985]),rz(-110))
    poses['right']=mix_pose(self.saved['pickup'],self.object_to_grip('right',handoff),(t-3)/2)
    openj['right']=.0015
    if t>=4:
        p,_=self.pose('head_grasp')
        body_rotation=self.data.body('lock_head').xmat.reshape(3,3)
        target=(p,body_rotation@display_head.T@self.reference.pose(12,'left')[1])
        poses['left']=mix_pose(self.home['left'],target,(t-4)/1.8)
        if t>=5.8:
            openj['left']=.0015
            if not self.grasped['left'] and t<6.5:self.capture('left')
        if self.grasped['left']:poses['left']=self.saved.setdefault('left_hold',self.pose('left_grip'))
    if t>=6.5:
        if not self.grasped['left'] and t<21.8:
            self.failure='Left head grasp missed';return poses,openj
        openj['left']=.0015
        if 'right_release_pose' not in self.saved:
            self.saved['right_release_pose']=self.pose('right_grip')
            head_rotation=self.reference.pose(12,'left')[1]@self.offsets['left'][1]
            self.saved['head_goal']=(np.array([.33,.045,1.025]),head_rotation)
            self.saved['head_before_regrasp']=(self.data.body('lock_head').xpos.copy(),self.data.body('lock_head').xmat.reshape(3,3).copy())
            # The left wrist turns the held head while the right hand is open,
            # as seen in the reference. This presents the free tail to the
            # measured low wrist regrasp pose, rather than reaching over it.
            setup_rotation=self.reference.pose(8,'right')[1]@self.reference.pose(12,'right')[1].T@head_rotation@hole_local@rz(180)
            self.saved['head_regrasp_pose']=(np.array([.30,.05,1.065]),setup_rotation)
            self.release('right','right_release_for_regrasp')
        if t<9.5:
            openj['right']=.025
            poses['left']=self.object_to_grip('left',mix_pose(self.saved['head_before_regrasp'],self.saved['head_regrasp_pose'],(t-6.5)/1.5))
            start=self.saved['right_release_pose']
            # Retarget the regrasp frame so the final wrist posture matches
            # the measured insertion posture despite the modeled grasp offset.
            tail_goal=self.saved['head_goal'][1]@hole_local@rz(180)
            current_tail=self.data.body('tail_body').xmat.reshape(3,3)
            rotation=current_tail@tail_goal.T@self.reference.pose(12,'right')[1]
            retreat=np.array([.255,-.035,.925])
            if t<7.3:
                poses['right']=mix_pose(start,(retreat,rotation),(t-6.5)/.8)
            else:
                point=self.pose('tail_grasp')[0]
                poses['right']=mix_pose((retreat,rotation),(point,rotation),(t-7.3)/1.2)
            if t>=8.5:
                openj['right']=.0015
                if not self.right_regrasped:self.capture('right')
                if self.right_regrasped:poses['right']=self.saved.setdefault('right_regrasp_pose',self.pose('right_grip'))
            return poses,openj
    if t>=9.5:
        if not self.right_regrasped:
            self.failure='Right tail regrasp missed';return poses,openj
        openj={'left':.0015,'right':.0015}
        if 'head_before_flip' not in self.saved:
            self.saved['head_before_flip']=(self.data.body('lock_head').xpos.copy(),self.data.body('lock_head').xmat.reshape(3,3).copy())
            self.saved['tail_before_flip']=(self.data.body('tail_body').xpos.copy(),self.data.body('tail_body').xmat.reshape(3,3).copy())
            # Preserve captured grasp transforms. Only physical robot motion
            # changes the strip shape; no scripted reorientation of lock anchors.
            head_rotation=self.reference.pose(12,'left')[1]@self.offsets['left'][1]
            self.saved['head_goal']=(np.array([.33,.045,1.025]),head_rotation)
        head=self.saved['head_goal']
        channel=head[1]@hole_local
        tail=(head[0]+channel[:,0]*.089,channel@rz(180))
        hpose=mix_pose(self.saved['head_before_flip'],head,(t-9.5)/4)
        tpose=mix_pose(self.saved['tail_before_flip'],tail,(t-9.5)/4)
        arc=np.sin(np.pi*float(smooth((t-9.5)/4)))
        tpose=(tpose[0]+np.array([.025,-.085,-.015])*arc,tpose[1])
        poses['left']=self.object_to_grip('left',hpose)
        poses['right']=self.object_to_grip('right',tpose)
        if t>=13.5:
            tail=(tail[0]-channel[:,0]*.058*smooth((t-13.5)/4),tail[1])
            poses['right']=self.object_to_grip('right',tail)
            if self.success and not self.inserted:self.latch()
        if t>=19:
            if not self.inserted:
                self.failure='Insertion was not verified';return poses,openj
            if self.grasped['right']:self.release('right','right_release_after_insertion')
            openj['right']=.025
            place_rotation=self.reference.pose(19,'left')[1]@self.offsets['left'][1]
            place=(np.array(self.layout['place_head']),place_rotation)
            poses['left']=self.object_to_grip('left',mix_pose(head,place,(t-19)/2.8))
            if t>=21.8:
                if not self.placed:
                    self.release('left','left_release_to_tray');self.placed=True
                openj['left']=.025
    return poses,openj

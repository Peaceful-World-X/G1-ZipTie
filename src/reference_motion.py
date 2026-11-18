"""Measured arm posture reference from episode 0 of the supplied Parquet file.

The measured joints guide retargeted IK; they are not claimed to be a calibrated
replay of the real workcell. Gripper channels are 0..100 closure percentages.
"""
import numpy as np
import mujoco
from simulation import ROOT

ARM_NAMES=('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')

class ReferenceMotion:
    def __init__(self,model):
        with np.load(ROOT/'references/recorded_joint_states.npz') as a:
            mask=a['episode']==0
            self.times=a['times'][mask].copy();self.states=a['state'][mask].copy()
        self.model=model;self.data=mujoco.MjData(model)
        self.arms={side:np.array([model.joint(f'{side}_{n}_joint').qposadr[0] for n in ARM_NAMES]) for side in ('left','right')}

    def joints(self,seconds,side):
        offset=0 if side=='left' else 7
        return np.array([np.interp(seconds,self.times,self.states[:,offset+i]) for i in range(7)])

    def pose(self,seconds,side):
        for s in ('left','right'):self.data.qpos[self.arms[s]]=self.joints(seconds,s)
        mujoco.mj_kinematics(self.model,self.data)
        site=self.data.site(side+'_grip')
        return site.xpos.copy(),site.xmat.reshape(3,3).copy()

    def mapped_time(self,simulation_time):
        return float(np.interp(simulation_time,[0,2.4,5.8,8.8,13.5,17.5,19,22,24],[0,1.8,5.5,8.5,12,16,17.5,19.2,20.733334]))

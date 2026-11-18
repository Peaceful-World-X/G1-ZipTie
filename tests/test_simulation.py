"""Behavior checks: physical rollout, open aperture, collision counterexample."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'src'))
import unittest
import numpy as np
import mujoco
from simulation import G1ZipTieEnv, DURATION, ROOT
from scipy.spatial.transform import Rotation

class TestInsertion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=G1ZipTieEnv()

    def test_01_actuated_rollout_and_reset(self):
        e=self.env; e.reset(); samples=[]
        while e.data.time<DURATION:
            e.step(dt=.02)
            self.assertTrue(np.isfinite(e.data.qpos).all())
            if e.data.time<9: self.assertFalse(e.success)
            samples.append(e.info())
        self.assertTrue(e.success)
        self.assertGreater(e.metrics()['depth_mm'],13)
        self.assertLess(e.metrics()['lateral_mm'],.5)
        self.assertLess(e.max_tail_contact,.0005)
        mask=e.model.jnt_limited.astype(bool)
        self.assertTrue(np.all(e.data.qpos[mask]>=e.model.jnt_range[mask,0]-.001))
        self.assertTrue(np.all(e.data.qpos[mask]<=e.model.jnt_range[mask,1]+.001))
        self.assertEqual(int(e.data.warning.number.sum()),0)
        e.reset(); self.assertFalse(e.success); self.assertEqual(e.data.time,0)

    def test_02_real_hole_and_collision_counterexample(self):
        e=self.env; e.reset()
        e.data.qpos[:]=e.plan[-1][1]
        mujoco.mj_forward(e.model,e.data)
        def contacts():
            return [c for c in e.data.contact if e.tailgeom in (c.geom1,c.geom2) and ({c.geom1,c.geom2}&e.lockgeoms)]
        self.assertEqual(len(contacts()),0,'Aligned tail must pass through the open hole')
        target=e.data.site_xpos[e.tip].copy(); target[0]+=.004
        q,error=e.solve_ik('right',target,Rotation.from_euler('z',90,degrees=True).as_matrix(),e.data.qpos)
        self.assertLess(error,.002)
        # Geometric fixture test only; live rollouts never rewrite joint positions.
        e.data.qpos[:]=q; mujoco.mj_forward(e.model,e.data)
        self.assertGreater(len(contacts()),0,'Offset tail must collide with a lock side wall')
        e.ever_entered=True; e.hold=1; e.update_success()
        self.assertFalse(e.success,'An off-axis tail must not count as an insertion')

    def test_03_action_validation(self):
        e=self.env; e.reset()
        with self.assertRaises(ValueError): e.step({'left_arm':[0]})
        with self.assertRaises(ValueError): e.step({'right_arm':[float('nan')]*7})

    def test_04_elastic_strip_responds_and_recovers(self):
        e=self.env; e.reset()
        while e.data.time<18: e.step(dt=.02)
        baseline=e.data.site_xpos[e.strip_sites].copy()
        energy=e.info()['strip_elastic_energy_mJ']
        e.perturb_strip(); peak=0.; energies=[]
        for i in range(300):
            e.step(dt=.01)
            displacement=np.linalg.norm(e.data.site_xpos[e.strip_sites]-baseline,axis=1).max()
            peak=max(peak,displacement); energies.append(e.info()['strip_elastic_energy_mJ'])
        final=np.linalg.norm(e.data.site_xpos[e.strip_sites]-baseline,axis=1).max()
        print(f'Elastic test: peak={peak*1000:.2f}mm, residual={final*1000:.2f}mm, energy range={max(energies)-min(energies):.5f}mJ',flush=True)
        self.assertGreater(peak,.001)
        self.assertLess(final,peak*.6)
        self.assertGreater(max(energies)-min(energies),.00001)
        self.assertLess(e.info()['strip_endpoint_error_mm'],.5)
        self.assertEqual(int(e.data.warning.number.sum()),0)

    def test_05_pose_blend_handles_float_endpoints(self):
        from task_sequence import mix_pose
        a=(np.zeros(3),np.eye(3));b=(np.ones(3),Rotation.from_euler('z',1).as_matrix())
        for t in (-.01,0.,1.-1e-9,1.-1e-12,np.nextafter(1.,0.),1.,1.01):
            p,R=mix_pose(a,b,t)
            self.assertTrue(np.all((p>=0)&(p<=1)))
            self.assertAlmostEqual(np.linalg.det(R),1.,places=10)

if __name__=='__main__': unittest.main(verbosity=2)

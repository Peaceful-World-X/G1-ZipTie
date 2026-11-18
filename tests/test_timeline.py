"""Restoring the controller and mutable constraints must match uninterrupted physics."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import mujoco
from replay import G1ZipTieEnv, DURATION, SCRIPT_RATE
from timeline import Timeline, STATE_SPEC
from simulation import projected_box_gap


class TestTimeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=G1ZipTieEnv()
        cls.timeline=Timeline(cls.env)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_seek_across_grasp_insert_release_events(self):
        e=self.env;t=self.timeline
        # Compare restored state followed by live physics against uninterrupted checkpoints.
        for before,after in [(2.1,2.5),(5.7,6.1),(6.4,6.9),(8.4,9.),(17.,18.5),(21.7,22.2)]:
            with self.subTest(before=before):
                t.seek(before)
                e.step(dt=after-before)
                observed=t.capture()
                t.seek(after)
                expected=t.capture()
                np.testing.assert_allclose(observed['states'],expected['states'],atol=2e-7,rtol=2e-7)
                self.assertEqual(observed['metadata'],expected['metadata'])

    def test_backwards_seek_clears_success_and_restores_constraints(self):
        e=self.env;t=self.timeline
        t.seek(DURATION)
        self.assertTrue(e.info()['task_complete'])
        e.perturb_strip()
        t.seek(0)
        self.assertFalse(e.inserted)
        self.assertFalse(e.placed)
        self.assertEqual(e.index,0)
        self.assertEqual(e.pulse_until,-1)
        self.assertFalse(e.data.eq_active[e.model.equality('thread_latch').id])
        self.assertFalse(e.data.eq_active[e.model.equality('right_grasp').id])
        self.assertIsNone(e.failure)

    def test_preview_exact_seek_and_bounds(self):
        e=self.env;t=self.timeline
        t.seek(18.253,preview=True)
        self.assertAlmostEqual(e.data.time,18.25,places=7)
        t.seek(18.253)
        self.assertAlmostEqual(e.data.time,18.253,places=7)
        t.seek(-2)
        self.assertEqual(e.data.time,0)
        t.seek(100)
        self.assertAlmostEqual(e.data.time,DURATION,places=7)
        self.assertTrue(e.info()['task_complete'])
        with self.assertRaises(ValueError):t.seek(float('nan'))

    def test_initial_arms_mirror_and_strip_has_straight_rest_shape(self):
        e=self.env;self.timeline.seek(0)
        mirror=np.array([1,-1,-1,1,-1,1,-1])
        np.testing.assert_allclose(e.data.qpos[e.arms['right']],e.data.qpos[e.arms['left']]*mirror,atol=1e-12)
        points=e.data.site_xpos[e.strip_sites]
        axis=points[-1]-points[0];axis/=np.linalg.norm(axis)
        deviations=np.linalg.norm(np.cross(points-points[0],axis),axis=1)
        self.assertLess(float(deviations.max()),1e-8)

    def test_horizontal_handoff_and_real_release_regrasp(self):
        e=self.env;t=self.timeline
        t.seek(5.5)
        self.assertTrue(e.grasped['right'])
        self.assertFalse(e.grasped['left'])
        delta=e.data.body('lock_head').xpos-e.data.body('tail_body').xpos
        tilt=np.degrees(np.arctan2(abs(delta[2]),np.linalg.norm(delta[:2])))
        self.assertLess(tilt,12,'Handoff must be horizontal, not a hanging vertical strip')
        t.seek(7.5)
        self.assertTrue(e.grasped['left']);self.assertFalse(e.grasped['right'])
        self.assertFalse(e.data.eq_active[e.model.equality('right_grasp').id])
        t.seek(9.2)
        self.assertTrue(e.grasped['left']);self.assertTrue(e.grasped['right'])

    def test_smooth_display_preserves_physics_checkpoints(self):
        e=self.env;t=self.timeline
        for seconds in (0.,5.5,7.5,9.2,18.,24.):
            t.seek(seconds);expected=e.data.qpos.copy();flags=(e.inserted,e.placed,dict(e.grasped))
            t.display(seconds)
            np.testing.assert_allclose(e.data.qpos,expected,atol=1e-10)
            self.assertEqual((e.inserted,e.placed,e.grasped),flags)
        t.display(5.555)
        self.assertAlmostEqual(e.data.time,5.555,places=8)
        self.assertTrue(np.isfinite(e.data.qpos).all())
        # Switching to live solver restores the exact baseline integration state.
        t.seek(5.555);e.step(dt=.005);actual=e.data.qpos.copy()
        t.seek(5.56)
        np.testing.assert_allclose(e.data.qpos,actual,atol=2e-7,rtol=2e-7)

    def test_lock_exposed_above_left_fingers_and_flip_is_continuous(self):
        e=self.env;t=self.timeline
        for seconds in (6.2,9.2,18.):
            t.seek(seconds)
            clearance=e.data.site('hole').xpos[2]-e.data.site('left_grip').xpos[2]
            self.assertGreater(clearance,.004,'Lock head must be exposed above the left gripping point')
        arms=np.r_[e.arms['left'],e.arms['right']]
        indices=[i for i,q in enumerate(e.ctrl_q) if q in arms]
        increments=np.max(np.abs(np.diff(e.cache['ctrl'][:,indices],axis=0)),axis=1)
        times=e.cache['times'][1:];flip=(times>=9.5)&(times<=13.6)
        self.assertLess(float(increments[flip].max()),.12,'Flip must not switch IK branches')

    def test_left_forearm_is_level_and_holds_downward_strip_still(self):
        e=self.env;t=self.timeline
        positions=[];rotations=[]
        for seconds in (8.4,9.2,11.5,13.8,18.):
            t.seek(seconds)
            grip=e.data.site('left_grip');rot=grip.xmat.reshape(3,3)
            self.assertGreater(rot[2,2],.97,'Left jaws must lie flat with the camera above them')
            v=e.data.body('left_wrist_yaw_link').xpos-e.data.body('left_elbow_link').xpos
            tilt=np.degrees(np.arctan2(abs(v[2]),np.linalg.norm(v[:2])))
            self.assertLess(tilt,20,'Left forearm must rest naturally across the workspace')
            positions.append(grip.xpos.copy());rotations.append(rot.copy())
        self.assertLess(np.ptp(np.array(positions),axis=0).max(),.001)
        self.assertLess(np.abs(np.array(rotations)-rotations[0]).max(),.01)
        indices=[i for i,q in enumerate(e.ctrl_q) if q in e.arms['left']]
        held=e.cache['ctrl'][(e.cache['times']>=8.4)&(e.cache['times']<19)][:,indices]
        self.assertLess(np.ptp(held,axis=0).max(),1e-10)

    def test_straight_strip_points_down_after_left_takeover(self):
        e=self.env
        for seconds in (8.4,8.65):
            self.timeline.seek(seconds)
            self.assertTrue(e.grasped['left']);self.assertFalse(e.grasped['right'])
            points=e.data.site_xpos[e.strip_sites]
            axis=points[-1]-points[0];axis/=np.linalg.norm(axis)
            tilt=np.degrees(np.arccos(np.clip(-axis[2],-1,1)))
            self.assertLess(tilt,5,'After takeover, the strip must point vertically down')
            deviations=np.linalg.norm(np.cross(points-points[0],axis),axis=1)
            self.assertLess(float(deviations.max()),.003,'The unsupported strip must remain straight')

    def test_tip_traverses_vertical_aperture_from_the_side(self):
        e = self.env
        samples = []
        for second in np.arange(12.8, 17.81, .05):
            self.timeline.display(second)
            hole = e.data.site('hole')
            basis = hole.xmat.reshape(3, 3)
            self.assertLess(abs(basis[2, 0]), np.sin(np.radians(2)))
            for wall in ('top', 'bottom', 'a', 'b'):
                wall_basis = e.data.geom('lock_' + wall).xmat.reshape(3, 3)
                np.testing.assert_allclose(wall_basis, basis, atol=1e-6)
            tip = e.data.site('tip').xpos.copy()
            local = basis.T @ (tip - hole.xpos)
            if abs(local[0]) <= .004:
                # The 3 x 0.8 mm tongue must fit inside the 5 x 2.4 mm bore.
                self.assertLess(abs(local[1]) + .0015, .0025)
                self.assertLess(abs(local[2]) + .0004, .0012)
            for part in ('neck_bridge', 'grasp_neck'):
                neck_axis = e.data.body('lock_head').xmat.reshape(3, 3)[:,0]
                gap = projected_box_gap(e, 'rigid_tail', part, neck_axis)
                self.assertGreater(gap, .001, 'The neck must leave the exit clear for the tongue')
            samples.append((local, tip, basis[:,0].copy()))
        local = np.array([row[0] for row in samples])
        self.assertGreater(local[0,0], .004)
        self.assertLess(local[-1,0], -.009)
        self.assertGreater(np.count_nonzero(abs(local[:,0]) <= .004), 5)
        travel = samples[-1][1] - samples[0][1]
        self.assertGreater(-travel @ samples[0][2], .03)
        self.assertLess(abs(travel[2]), .0015)

    def test_moulded_buckle_has_an_open_visual_channel(self):
        e = self.env
        self.timeline.display(15.4)
        mesh_id = e.model.mesh('moulded_buckle').id
        geoms = np.flatnonzero((e.model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
                               & (e.model.geom_dataid == mesh_id))
        self.assertEqual(len(geoms), 1)
        geom = int(geoms[0])
        self.assertEqual(e.model.geom_contype[geom] | e.model.geom_conaffinity[geom], 0)
        hole = e.data.site('hole')
        basis = hole.xmat.reshape(3, 3)
        for direction in (-1, 1):
            axis = direction * basis[:,0]
            origin = hole.xpos + .015 * axis
            for across, height in ((0., 0.), (.0021, .00085), (-.0021, -.00085)):
                ray = origin + across*basis[:,1] + height*basis[:,2]
                self.assertLess(mujoco.mj_rayMesh(e.model, e.data, geom, ray, -axis), 0)
            rim = origin + .0038 * basis[:,1]
            self.assertAlmostEqual(mujoco.mj_rayMesh(e.model, e.data, geom, rim, -axis), .011, places=4)

    def test_wrist_cameras_follow_the_grippers_through_the_flip(self):
        e=self.env
        for t,sign in ((0,1),(5.5,1),(9.2,-1),(12.8,1),(17.7,1)):
            self.timeline.seek(t)
            for side in ('left','right'):
                cam=e.data.camera(side+'_wrist_cam')
                def project(points):
                    local = (points-cam.xpos) @ cam.xmat.reshape(3,3)
                    half_height = np.tan(np.radians(e.model.cam_fovy[cam.id])/2)
                    assert np.all(local[...,2] < 0)
                    return .5 + local[...,:2] / (-2*local[...,2,None]*half_height) * np.array([.75,-1])
                grip = project(e.data.site(side+'_grip').xpos)
                self.assertTrue(.30 < grip[0] < .70 and .49 < grip[1] < .75, (side, t, grip))
                if t == 17.7:
                    lock = project(e.data.site('hole').xpos)
                    self.assertTrue(np.all((lock > .10) & (lock < .90)), (side, lock))
                    if side == 'right':
                        loop = project(e.data.site_xpos[e.strip_sites])
                        self.assertTrue(np.all((loop > .03) & (loop < .99)), loop)
                        self.assertGreater(loop[:,0].max()-lock[0], .12)
                offset=e.data.body(side+'_d405').xipos-e.data.body(side+'_tool').xpos
                self.assertGreater(offset[2]*(sign if side=='right' else 1),.025)
            if t>=9.2:
                right_axis=e.data.site('right_grip').xmat.reshape(3,3)[:,0]
                self.assertGreater(right_axis[1],.8,'Right fingers approach across the center lane')


if __name__=='__main__':unittest.main()

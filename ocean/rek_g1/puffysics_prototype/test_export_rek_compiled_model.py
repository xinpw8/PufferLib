import copy
import unittest

import numpy as np

import export_rek_compiled_model as adapter

try:
    import mujoco
except ImportError:
    mujoco = None


class FrameMathTests(unittest.TestCase):
    def test_quaternion_matrix_roundtrip(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            r = adapter.quat_matrix(q)
            np.testing.assert_allclose(adapter.quat_matrix(adapter.matrix_quat(r)), r, atol=1e-12)

    def test_capsule_y_endpoints_preserve_source_z(self):
        q = np.array([.2, -.3, .4, .5])
        q /= np.linalg.norm(q)
        r = adapter.quat_matrix(q)
        p = np.array([.4, -.8, 1.2])
        tp, tr = adapter.capsule_y_pose(p, r)
        for sign in (-1, 1):
            np.testing.assert_allclose(p+r@np.array([0, 0, sign*.7]),
                                       tp+tr@np.array([0, sign*.7, 0]), atol=1e-12)

    def test_relative_pose_roundtrip(self):
        parent = adapter.quat_matrix([.5, .5, .5, .5])
        rotation = adapter.quat_matrix([0, 0, 1, 0])
        pp, p = np.array([2., 3., 4.]), np.array([-1., 4., .2])
        lp, lr = adapter.relative_pose(pp, parent, p, rotation)
        np.testing.assert_allclose(pp+parent@lp, p, atol=1e-12)
        np.testing.assert_allclose(parent@lr, rotation, atol=1e-12)

    def test_reference_tangent_is_unit_and_perpendicular(self):
        for axis in np.eye(3).tolist()+[[.6, 0, .8]]:
            tangent = adapter.reference_tangent(axis)
            self.assertAlmostEqual(np.linalg.norm(tangent), 1.)
            self.assertAlmostEqual(np.dot(tangent, axis), 0.)


@unittest.skipIf(mujoco is None, "CPU MuJoCo required for compiled integration tests")
class CompiledAdapterTests(unittest.TestCase):
    def make_model(self):
        # A massive free body and hinged child have intentionally rotated inertias.
        xml = '''<mujoco><option gravity="0 0 -9.81"/><worldbody>
        <geom type="box" size="2 2 .1" pos="0 0 -.1"/>
        <body pos="0 0 1" quat=".9238795325 0 0 .3826834324"><freejoint/>
          <inertial pos=".1 .02 -.03" quat=".9238795325 .3826834324 0 0" mass="2" diaginertia=".2 .3 .4"/>
          <geom type="capsule" size=".05 .2" pos=".1 0 .2"/>
          <body pos="0 .1 .2"><geom type="sphere" size=".03" mass="0"/></body>
          <body pos="0 0 .4" quat=".9238795325 0 .3826834324 0">
            <joint name="hinge" axis="0 1 0" limited="true" range="-60 90" armature=".03" damping=".05" frictionloss=".1"/>
            <inertial pos="-.03 .04 .1" quat=".9238795325 0 0 .3826834324" mass="1" diaginertia=".1 .12 .13"/>
            <geom type="cylinder" size=".04 .12" pos="0 0 .1"/>
          </body>
        </body></worldbody><actuator><motor joint="hinge"/></actuator></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        data.qpos[-1] = .21
        mujoco.mj_kinematics(model, data)
        return model, data

    def export(self):
        model, data = self.make_model()
        out = adapter.export_compiled(model, data, mujoco, source={}, initial_pose={"kind": "test"})
        return model, data, out

    def test_compiled_counts_and_massless_helper_transfer(self):
        model, data, out = self.export()
        self.assertEqual(out['counts']['rigid_bodies'], 2)
        self.assertEqual(out['counts']['shapes'], 4)
        self.assertEqual(out['counts']['hinges'], 1)
        self.assertEqual(out['counts']['free_bases'], 1)
        sphere = next(shape for shape in out['shapes'] if shape['kind'] == 'sphere')
        self.assertEqual(sphere['rigid_body_id'], 0)
        self.assertEqual(out['shapes'][0]['rigid_body_id'], -1)
        self.assertEqual(out['initial_qpos'][-1], .21)

    def test_every_source_shape_world_pose_reconstructs(self):
        model, data, out = self.export()
        for shape in out['shapes']:
            owner = shape['rigid_body_id']
            if owner < 0:
                p, r = np.zeros(3), np.eye(3)
            else:
                pose = out['bodies'][owner]['world_pose']
                p = np.array(pose['position_xyz'])
                r = adapter.quat_matrix(pose['quaternion_xyzw'])
            local = shape['source_local_pose']
            np.testing.assert_allclose(p+r@local['position_xyz'], data.geom_xpos[shape['id']], atol=1e-10)
            np.testing.assert_allclose(r@adapter.quat_matrix(local['quaternion_xyzw']),
                                       data.geom_xmat[shape['id']].reshape(3, 3), atol=1e-10)

    def test_original_link_world_pose_reconstructs(self):
        model, data, out = self.export()
        for body in out['bodies']:
            w, l = body['world_pose'], body['original_link_in_rigid']
            r = adapter.quat_matrix(w['quaternion_xyzw'])
            np.testing.assert_allclose(np.array(w['position_xyz'])+r@l['position_xyz'],
                                       data.xpos[body['source_body_id']], atol=1e-10)
            np.testing.assert_allclose(r@adapter.quat_matrix(l['quaternion_xyzw']),
                                       data.xmat[body['source_body_id']].reshape(3, 3), atol=1e-10)

    def test_capsule_and_cylinder_target_axis_preserved(self):
        model, data, out = self.export()
        for shape in out['shapes']:
            if shape['kind'] not in ('capsule', 'cylinder'):
                continue
            source = adapter.quat_matrix(shape['source_local_pose']['quaternion_xyzw'])
            target = adapter.quat_matrix(shape['target_local_pose']['quaternion_xyzw'])
            np.testing.assert_allclose(source[:, 2], target[:, 1], atol=1e-12)
            self.assertEqual(shape['target_long_axis'], 'y')

    def test_joint_offset_and_axes_match(self):
        model, data, out = self.export()
        hinge = out['hinges'][0]
        self.assertAlmostEqual(hinge['mujoco_initial_angle_rad'], .21)
        self.assertEqual(hinge['qposadr'], model.nq-1)
        self.assertEqual(hinge['dofadr'], model.nv-1)
        np.testing.assert_allclose(hinge['limits_relative_to_initial_rad'], model.jnt_range[-1]-.21)
        adapter.validate_export(out)
        broken = copy.deepcopy(out)
        broken['hinges'][0]['anchor_child_xyz'][0] += .01
        with self.assertRaisesRegex(ValueError, 'endpoint frame'):
            adapter.validate_export(broken)

    def test_inertia_is_principal_and_preserved(self):
        model, data, out = self.export()
        for body in out['bodies']:
            np.testing.assert_array_equal(body['principal_inertia_kg_m2'], model.body_inertia[body['source_body_id']])
        self.assertEqual(out['execution'], {'physics_steps': 0, 'gpu_initializations': 0})

    def test_unsupported_joint_fails_without_dropping(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint type="slide"/><geom size=".1"/></body></worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        with self.assertRaisesRegex(ValueError, 'joint type'):
            adapter.export_compiled(model, data, mujoco, source={}, initial_pose={})


if __name__ == '__main__':
    unittest.main()

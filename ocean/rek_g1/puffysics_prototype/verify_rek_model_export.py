"""Check every exported initial transform against the pinned compiled model."""
import json
from pathlib import Path
import sys

import numpy as np
import mujoco

import export_rek_compiled_model as adapter


def verify(path):
    record = json.loads(Path(path).read_text())
    adapter.validate_export(record)
    source = Path(record['source']['model_path'])
    assert adapter.file_digest(source) == record['source']['model_sha256']
    model = mujoco.MjModel.from_xml_string(source.read_text())
    data = mujoco.MjData(model)
    metadata = record['initial_pose']
    initial, _ = adapter.idle_qpos(model, Path(metadata['manifest_path']).parent,
                                   metadata['manifest_sha256'])
    np.testing.assert_array_equal(initial, record['initial_qpos'])
    data.qpos[:] = initial
    mujoco.mj_kinematics(model, data)
    maxima = dict(body_position=0., body_rotation=0., shape_position=0., shape_rotation=0.)
    world = {-1: (np.zeros(3), np.eye(3))}
    for body in record['bodies']:
        rigid = body['id']
        pose = body['world_pose']
        p = np.array(pose['position_xyz'])
        r = adapter.quat_matrix(pose['quaternion_xyzw'])
        world[rigid] = p, r
        local = body['original_link_in_rigid']
        body_id = body['source_body_id']
        dp = np.max(np.abs(p+r@local['position_xyz']-data.xpos[body_id]))
        dr = np.max(np.abs(r@adapter.quat_matrix(local['quaternion_xyzw'])-data.xmat[body_id].reshape(3, 3)))
        maxima['body_position'] = max(maxima['body_position'], float(dp))
        maxima['body_rotation'] = max(maxima['body_rotation'], float(dr))
        np.testing.assert_array_equal(body['principal_inertia_kg_m2'], model.body_inertia[body_id])
        assert body['mass_kg'] == model.body_mass[body_id]
    assert [item['source_geom_id'] for item in record['shapes']] == list(range(model.ngeom))
    expected_hinges = np.flatnonzero(model.jnt_type == int(mujoco.mjtJoint.mjJNT_HINGE)).tolist()
    assert [item['source_joint_id'] for item in record['hinges']] == expected_hinges
    for shape in record['shapes']:
        p, r = world[shape['rigid_body_id']]
        pose = shape['target_local_pose']
        position = p+r@pose['position_xyz']
        rotation = r@adapter.quat_matrix(pose['quaternion_xyzw'])
        geom = shape['source_geom_id']
        desired = data.geom_xmat[geom].reshape(3, 3)
        if shape['kind'] in ('capsule', 'cylinder'):
            _, desired = adapter.capsule_y_pose(np.zeros(3), desired)
        maxima['shape_position'] = max(maxima['shape_position'], float(np.max(np.abs(position-data.geom_xpos[geom]))))
        maxima['shape_rotation'] = max(maxima['shape_rotation'], float(np.max(np.abs(rotation-desired))))
    assert max(maxima.values()) <= 1e-9, maxima
    return dict(schema='rek.puffysics.adapter_initial_pose_verification.v1',
                adapter_sha256=adapter.file_digest(path), counts=record['counts'],
                exact_idle_qpos_match=True, all_shapes_and_hinges_preserved=True,
                mass_and_principal_inertia_exact=True, maximum_absolute_errors=maxima,
                comparison_tolerance=1e-9, physics_steps=0, gpu_initializations=0)


if __name__ == '__main__':
    print(json.dumps(verify(sys.argv[1]), indent=2))

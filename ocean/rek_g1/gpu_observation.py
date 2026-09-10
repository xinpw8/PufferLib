"""CUDA assembly of the existing 223-float semantic duel observation ABI."""

from __future__ import annotations

import numpy as np
import torch


def inertial_body_velocity(cvel, ximat, xipos, root_com):
    """Match mj_objectVelocity(mjOBJ_BODY, local=1), without host reads.

    MuJoCo BODY means the inertial frame. XBODY instead uses xmat/xpos.
    cvel is centered on the kinematic tree's subtree center of mass.
    """
    device = cvel.device
    if device.type != "cuda" or any(
        value.device != device for value in (ximat, xipos, root_com)
    ):
        raise ValueError("body velocity operands must share a CUDA device")
    angular = cvel[..., :3].double()
    lever = xipos.double() - root_com.double()
    linear = cvel[..., 3:].double() + torch.linalg.cross(angular, lever, dim=-1)
    inverse_rotation = ximat.double().transpose(-1, -2)
    angular_local = torch.matmul(inverse_rotation, angular[..., None]).squeeze(-1)
    linear_local = torch.matmul(inverse_rotation, linear[..., None]).squeeze(-1)
    return torch.cat((angular_local, linear_local), dim=-1)


class GpuObservationAssembler:
    """Gather physical fields and pack supplied measured/native state rows.

    Fall, motion and fight fields are required inputs. This module supplies no
    defaults or inferred combat outcomes when one of those subsystems is absent.
    """

    def __init__(self, physics):
        self.physics = physics
        self.rows = physics.arenas * 2
        self.device = physics.qpos.device
        tensor = lambda value: torch.as_tensor(value, device=self.device)
        self.root_bodies = tensor(physics.root_bodies)
        self.root_com_bodies = tensor(
            physics.host_model.body_rootid[physics.root_bodies].copy(),
        )
        self.qindices = tensor(np.stack(physics.joint_qpos))
        self.dqindices = tensor(np.stack(physics.joint_qvel))
        for name in ("cvel", "ximat", "xipos", "subtree_com"):
            setattr(self, name, physics.wp.to_torch(getattr(physics.data, name)))
        self.entities = torch.empty((self.rows, 86), device=self.device)
        self.observations = torch.empty((self.rows, 223), device=self.device)

    def gather_kinematics(self):
        physics = self.physics
        roots = physics.qpos.reshape(-1, 2, 36)[..., :7].reshape(self.rows, 7)
        velocity = inertial_body_velocity(
            self.cvel[:, self.root_bodies],
            self.ximat[:, self.root_bodies].reshape(-1, 2, 3, 3),
            self.xipos[:, self.root_bodies],
            self.subtree_com[:, self.root_com_bodies],
        ).reshape(self.rows, 6)
        self.entities[:, :7] = roots
        self.entities[:, 7:10] = velocity[:, 3:]
        self.entities[:, 10:13] = velocity[:, :3]
        self.entities[:, 13:42] = physics.qpos[:, self.qindices].reshape(self.rows, 29)
        self.entities[:, 42:71] = physics.qvel[:, self.dqindices].reshape(self.rows, 29)
        return self.entities[:, :71]

    def pack(self, fall, semantic, fight):
        for value, width, name in ((fall, 15, "fall"), (semantic, 12, "semantic"), (fight, 39, "fight")):
            if value.device != self.device or value.dtype != torch.float32:
                raise ValueError(f"{name} must be CUDA float32 on the physics device")
            if value.shape != (self.rows, width):
                raise ValueError(f"{name} observation shape mismatch")
        self.entities[:, 71:86] = fall
        self.observations[:, :86] = self.entities
        self.observations[:, 86:172] = self.entities.reshape(-1, 2, 86).flip(1).reshape(self.rows, 86)
        self.observations[:, 172:184] = semantic
        self.observations[:, 184:223] = fight
        return self.observations

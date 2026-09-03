import unittest

from mujoco_arena import transform_matrix, transform_point


class ArenaTransformTests(unittest.TestCase):
    def test_identity_point(self):
        chain = [{
            "local_position": {"x": 0, "y": 0, "z": 0},
            "local_rotation": {"w": 1, "x": 0, "y": 0, "z": 0},
            "local_scale": {"x": 1, "y": 1, "z": 1},
        }]
        self.assertEqual(transform_point(transform_matrix(chain), (1, 2, 3)), (1, 2, 3))

    def test_parent_translation_and_scale(self):
        chain = [
            {
                "local_position": {"x": 1, "y": 2, "z": 3},
                "local_rotation": {"w": 1, "x": 0, "y": 0, "z": 0},
                "local_scale": {"x": 2, "y": 3, "z": 4},
            },
            {
                "local_position": {"x": 1, "y": 1, "z": 1},
                "local_rotation": {"w": 1, "x": 0, "y": 0, "z": 0},
                "local_scale": {"x": 1, "y": 1, "z": 1},
            },
        ]
        self.assertEqual(transform_point(transform_matrix(chain), (0, 0, 0)), (3, 5, 7))


if __name__ == "__main__":
    unittest.main()

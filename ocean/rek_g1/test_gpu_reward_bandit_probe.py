import unittest

import torch

from gpu_reward_bandit_probe import RewardBandit, bandit_config


class RewardBanditTests(unittest.TestCase):
    def test_reward_and_terminal_alignment(self):
        env = RewardBandit(4, device="cpu")
        self.assertEqual(env.terminals.tolist(), [0, 0, 0, 0])
        env.step(torch.tensor([[0], [1], [0], [1]]))
        self.assertEqual(env.rewards.tolist(), [1, -1, 1, -1])
        self.assertEqual(env.terminals.tolist(), [1, 1, 1, 1])
        self.assertEqual(env.snapshot()["rewarded_action_fraction"], .5)
        self.assertEqual(env.snapshot()["mean_reward"], 0)
        self.assertEqual(env.action_mask.sum().item(), 8)

    def test_invalid_actions_are_visible(self):
        env = RewardBandit(2, device="cpu")
        env.step(torch.tensor([[0], [9]]))
        self.assertEqual(env.snapshot()["invalid_actions"], 1)

    def test_config_matches_native_update_count(self):
        native = {"vec": {}, "train": {"vf_coef": .5, "beta1": .9}}
        configured = bandit_config(native, updates=256, learning_rate=.0003)
        self.assertEqual(configured["train"]["total_timesteps"], 524288)
        self.assertEqual(configured["train"]["minibatch_size"], 2048)
        self.assertEqual(configured["train"]["learning_rate"], .0003)
        self.assertFalse(configured["train"]["anneal_lr"])
        self.assertEqual(configured["train"]["vf_coef"], .5)
        self.assertNotIn("total_timesteps", native["train"])
        repeated = bandit_config(native, updates=256, learning_rate=.003, replay_ratio=4)
        self.assertEqual(repeated["train"]["total_timesteps"], 524288)
        self.assertEqual(repeated["train"]["replay_ratio"], 4)

    def test_positive_actor_advantage_increases_rewarded_logit(self):
        # CPU autograd sanity check of the native discrete PPO gradient sign.
        logits = torch.tensor([.2, -.3], dtype=torch.float64, requires_grad=True)
        probabilities = logits.softmax(0)
        loss = -torch.log(probabilities[0]) * 1.5
        actual, = torch.autograd.grad(loss, logits)
        expected = -1.5 * (torch.tensor([1., 0.], dtype=torch.float64) - probabilities.detach())
        torch.testing.assert_close(actual, expected)
        self.assertLess(actual[0].item(), 0)
        self.assertGreater(actual[1].item(), 0)


if __name__ == "__main__":
    unittest.main()

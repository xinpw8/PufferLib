# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass
from copy import deepcopy

from tqdm import tqdm

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import gym as old_gym
import shimmy

import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.vector
import pufferlib.environments.trash_pickup


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "TrashPickup"
    """the id of the environment"""
    total_timesteps: int = 1000000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, emulated):
        super().__init__()
        self.dtype = pufferlib.pytorch.nativize_dtype(emulated)

        # Define network for processing the grid with multiple channels (4 channels for 0, 1, 2, 3)
        self.grid_net = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 3, stride=1)),  # Now assuming grid has 4 channels
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Linear layers for agent position and carrying status
        self.position_net = nn.Sequential(
            layer_init(nn.Linear(2, 32)),
            nn.ReLU(),
        )
        
        self.carrying_net = nn.Sequential(
            layer_init(nn.Linear(1, 32)),
            nn.ReLU(),
        )

        # Combine outputs
        self.proj = nn.Linear(2368, 256)  # (2368) hard-coded for now, fix later Adjusted projection layer size based on concatenated features
        self.actor = layer_init(nn.Linear(256, 4), std=0.01)  # 4 is hard-coded for now (represents number of actions)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def preprocess_grid(self, grid):
        # Separate grid into 4 channels for empty space, trash, bin, and agent
        empty_space = (grid == 0).float()
        trash = (grid == 1).float()
        trash_bin = (grid == 2).float()
        agent = (grid == 3).float()
        
        # Stack channels to create a (4, H, W) tensor
        ret = torch.stack([empty_space, trash, trash_bin, agent], dim=1)
        return ret

    def hidden(self, x):
        x = x.type(torch.uint8)  # Undo bad cleanrl cast
        x = pufferlib.pytorch.nativize_tensor(x, self.dtype)

        # Process each part of the observation
        grid = self.preprocess_grid(x['grid'])
        grid_features = self.grid_net(grid)

        position = x['agent_position'].float() / 10 # divide by 10 (grid-size) hard-coded for now... 
        position_features = self.position_net(position)

        carrying = x['carrying_trash'].float()
        carrying_features = self.carrying_net(carrying)

        # Concatenate all features and project to a fixed-size hidden representation
        concat = torch.cat([grid_features, position_features, carrying_features], dim=1)
        return self.proj(concat)

    def get_value(self, x):
        hidden = self.hidden(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self.hidden(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    torch.set_printoptions(threshold=torch.inf)

    # PufferLib vectorization
    envs = pufferlib.vector.make(
        pufferlib.environments.trash_pickup.env_creator(),
        backend=pufferlib.vector.Multiprocessing,
        num_envs=args.num_envs,
        num_workers=args.num_envs,  # // 2
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # ALGO Logic: Storage setup
    num_agents = 3 # hard-coded for now, adjust this later
    obs = torch.zeros((args.num_steps, args.num_envs, num_agents) + envs.single_observation_space.shape).to(device) 
    actions = torch.zeros((args.num_steps, args.num_envs, num_agents) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs, num_agents)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs, num_agents)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs, num_agents)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs, num_agents)).to(device)

    # dummy_obs = torch.zeros((args.num_steps, envs.single_observation_space.shape[0])).to(device)

    # Annoyance: AsyncVectorEnv does not have a driver env
    agent = Agent(envs.driver_env.emulated).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed) # returns tensor of shape ([6, 120]), but we need it of shape ([3, 2, 120])
    next_done = torch.zeros((args.num_envs, num_agents)).to(device)

    for iteration in tqdm(range(1, args.num_iterations + 1)):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs * num_agents
            obs[step] = torch.Tensor(next_obs.reshape(args.num_envs, num_agents, envs.single_observation_space.shape[0])).to(device) # error here
            dones[step] = torch.Tensor(next_done.reshape(args.num_envs, num_agents)).to(device)

            # dummy = next_obs.reshape(args.num_envs, num_agents, envs.single_observation_space.shape[0])
            # dummy_obs[step] = torch.Tensor(dummy[0][0]).to(device)

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(torch.Tensor(next_obs).to(device))
                values[step] = value.reshape(args.num_envs, num_agents)
            actions[step] = action.reshape(args.num_envs, num_agents)
            logprobs[step] = logprob.reshape(args.num_envs, num_agents)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(args.num_envs, num_agents)
            next_obs = torch.as_tensor(next_obs, device=device)
            next_done = torch.as_tensor(next_done, dtype=torch.float32, device=device)
            
            writer.add_scalar(f"charts/mean-reward", rewards[step].mean(), global_step)

            episode_reward_list = []
            episode_length_list = []
            trash_remaining_list = []
            for info in infos:               
                if "final_info" in info:
                    episode_reward_list.append(info["final_info"]["episode_reward"])
                    episode_length_list.append(info["final_info"]["episode_length"])
                    trash_remaining_list.append(info["final_info"]["trash_remaining"])

            if len(episode_reward_list) > 0:
                writer.add_scalar("charts/episodic_reward", np.mean(episode_reward_list), global_step)

            if len(episode_length_list) > 0:
                writer.add_scalar("charts/episodic_length", np.mean(episode_length_list), global_step)

            if len(trash_remaining_list) > 0:
                writer.add_scalar("charts/total_trash_remaining", np.mean(trash_remaining_list), global_step)


        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1].reshape(args.num_envs * num_agents)
                    nextvalues = values[t + 1].reshape(args.num_envs * num_agents)
                delta = rewards[t].reshape(args.num_envs * num_agents) + args.gamma * nextvalues * nextnonterminal - values[t].reshape(args.num_envs * num_agents)
                if not isinstance(lastgaelam, int):
                    lastgaelam = lastgaelam.reshape(args.num_envs * num_agents)
                advantages[t] = lastgaelam = (delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam).reshape(args.num_envs, num_agents)
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # if iteration % 5 == 0:
            # print(f"SPS: {int(global_step / (time.time() - start_time))} | Current Iteration: {iteration} out of {args.num_iterations}")
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

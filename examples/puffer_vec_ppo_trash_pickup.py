# Standard libraries for file operations, randomization, time management, and data handling
import os
import sys
import random
import time
from dataclasses import dataclass
from copy import deepcopy

# Libraries for progress tracking, environments, and RL model components
from tqdm import tqdm
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# PufferLib modules for environment emulation and agent training in a multi-agent setting
import pufferlib
import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.vector
import pufferlib.environments.trash_pickup


@dataclass
class Args:
     # Experiment configuration details, seed values, and directory paths
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 3
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
    run_mode: str = "train"
    """'train' to train a new model and save it, 'evaluate' to load an existing model and test performance, 'video' to load an existing model and create a video"""
    model_save_dir: str = None # '/'
    """Path to save the model to this directory"""
    model_save_filename: str = None # "trash_pickup_model"
    """Filename to save the model as (file extension is automatically added, do NOT include it)"""
    model_load_path: str = None # 'examples/trash_pickup_saves/trash_pickup_model_with_cnn_iter_1907.pt'
    """Path to load in existing model to continue training, perform evaluate, or create a video"""
    model_save_interval: float = 3
    """How (roughly) often to save the model in minutes"""
    num_eval_video_episodes: int = 5
    """Number of evaluation or video episodes to run (only for run_mode == 'evaluate' or 'video')"""

    # PPO algorithm-specific hyperparameters like learning rate, number of timesteps, etc.
    env_id: str = "TrashPickup"
    """the id of the environment"""
    total_timesteps: int = 250_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 8192
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
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
    ent_coef_decay: bool = False
    """Decays the entropy coefficient (ent_coef) from its starting value to 0 by the end of training"""

    # Environment-specific parameters, such as grid size, agent count, and max steps per episode
    num_agents: int = 4
    """The number of agents in the environment"""
    grid_size: int = 10
    """The square size of the environment"""
    num_trash: int = 15
    """The amount of 'trash' to be picked up by the agents in the environment"""
    num_bins: int = 1
    """The number of 'trash bins' to put 'trash' picked up by the agents into in the environment"""
    num_max_env_steps: int = 300
    """The maximum number of steps that can be taken in the environment before the episode automatically ends"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # Initialize the layer with orthogonal weights and set bias to a constant
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# For use with PPO controlling multiple agents, could be improved by filtering out 
# duplicate observations from each agent (trash and trash bin positions, and 
# 'other_agent_positions' & 'other_agent_carrying_trash' since agent_positions & carrying 
# for each agent already contains this data)
class Agent(nn.Module):
    def __init__(self, emulated):
        super().__init__()
        self.dtype = pufferlib.pytorch.nativize_dtype(emulated)

        # Trash positions 
        self.trash_net = nn.Sequential(
            layer_init(nn.Linear(3 * args.num_trash, 32)),  # (presence, x, y) for each trash
            nn.ReLU(),
            layer_init(nn.Linear(32, 16)),
            nn.ReLU()
        )

        # Bin positions
        self.bin_net = nn.Sequential(
            layer_init(nn.Linear(2 * args.num_bins, 8)),  # (x, y) for each bin
            nn.ReLU(),
            layer_init(nn.Linear(8, 4)),
            nn.ReLU()
        )

        # Other agents' positions and carrying status
        self.other_agent_net = nn.Sequential(
            layer_init(nn.Linear(3 * (args.num_agents - 1), 16)),  # (x, y, carrying) for each other agent
            nn.ReLU(),
            layer_init(nn.Linear(16, 8)),
            nn.ReLU()
        )

        # Current agent's position and carrying status
        self.position_net = nn.Sequential(
            layer_init(nn.Linear(2, 8)),
            nn.ReLU(),
        )
        
        self.carrying_net = nn.Sequential(
            layer_init(nn.Linear(1, 2)),
            nn.ReLU(),
        )

        # Projection layer to combine features from different sources
        self.proj = nn.Sequential( 
            nn.Linear(16 + 4 + 8 + 8 + 2, 32),  # Adjusted projection layer size based on concatenated features
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(16, 8)),
            nn.ReLU(),
            layer_init(nn.Linear(8, 4), std=0.01) # 4 represents the number of possible actions
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(16, 8)),
            nn.ReLU(),
            layer_init(nn.Linear(8, 1), std=1) # 1 represents the value of the critic network (or sometimes called value network) 
        )  

    def preprocess_observation(self, obs):
        # Extract trash positions
        trash_data = obs['trash_positions']  # Shape: [batch_size, num_trash, 3]
        trash_data = trash_data.float()  # Ensure it's a float tensor
        trash_data[:, :, 1:] /= args.grid_size  # Normalize (x, y) positions by grid size

        # Get batch size from trash_data to ensure consistency
        batch_size = trash_data.size(0)

        # Process bin positions and normalize
        bin_positions = obs['bin_positions'].view(batch_size, -1).float() / args.grid_size  # Normalize positions

        # Combine other agent positions and carrying status
        other_agent_positions = obs['other_agent_positions'].float() / args.grid_size  # Shape: [batch_size, num_agents - 1, 2]
        other_agent_carrying = obs['other_agent_carrying_trash'].float().unsqueeze(-1)  # Shape: [batch_size, num_agents - 1, 1]
        other_agents = torch.cat([other_agent_positions, other_agent_carrying], dim=-1)  # Shape: [batch_size, num_agents - 1, 3]
        other_agents = other_agents.view(batch_size, -1)  # Shape: [batch_size, (num_agents - 1) * 3]

        # Return preprocessed observations
        return {
            'trash_positions': trash_data.view(batch_size, -1),  # Flatten trash data for input to network
            'bin_positions': bin_positions,
            'other_agent_data': other_agents,
            'agent_position': obs['agent_position'].float() / args.grid_size,
            'carrying_trash': obs['carrying_trash'].float()
        }


    def hidden(self, x):
        x = x.type(torch.uint8)  # Undo bad cleanrl cast
        x = pufferlib.pytorch.nativize_tensor(x, self.dtype)

        # Preprocess observation
        obs = self.preprocess_observation(x)

        # Conditional forward pass for trash using requires_grad
        # if obs['trash_positions'].requires_grad:
        trash_features = self.trash_net(obs['trash_positions'])
        # else:
        #     trash_features = torch.zeros(16, device=device)  # 16 is the output dimension of trash_net

        bin_features = self.bin_net(obs['bin_positions'])
        other_agent_features = self.other_agent_net(obs['other_agent_data'])
        position_features = self.position_net(obs['agent_position'])
        carrying_features = self.carrying_net(obs['carrying_trash'])

        # Concatenate all features and project to a fixed-size hidden representation
        concat = torch.cat([trash_features, bin_features, other_agent_features, position_features, carrying_features], dim=1)
        proj = self.proj(concat)
        return proj

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
    
    def forward(self, x, state=None):
        action, log_prob, entropy, value = self.get_action_and_value(x)
        return action, log_prob, entropy, value


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Verify that the directory for saving models exists, otherwise exit
    if args.model_save_dir is not None and not os.path.isdir(args.model_save_dir):
        print(f"Model Save Directory [{args.model_save_dir}] does not exist, exitting...")
        print(f"Current absolute path is: {os.path.abspath('./')}")
        exit(0)
    elif args.run_mode == 'train' and args.model_save_dir is None:
        print(f"WARNING: Training but save location not specified. Continuing running to store training data, but model will not be saved")
    elif (args.run_mode == 'evaluate' or args.run_mode == 'video') and args.model_load_path is None:
        print(f"Can not run program with run_mode '{args.run_mode}' and args.model_load_path is None.")
        exit(0)
    
    batch_size = int(args.num_envs * args.num_steps)
    minibatch_size = int(batch_size // args.num_minibatches)
    num_iterations = args.total_timesteps // batch_size
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
        
    if args.run_mode == "train":
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

    num_agents = args.num_agents

    if args.run_mode == "video" and args.num_envs != 1:
        print(f"run_mode set to video, but num_envs is {args.num_envs}, changing num_envs to just 1")
        args.num_envs = 1

    # PufferLib vectorization
    envs = pufferlib.vector.make(
        pufferlib.environments.trash_pickup.env_creator(
            grid_size=args.grid_size,
            num_agents=num_agents,
            num_trash=args.num_trash,
            num_bins=args.num_bins,
            max_steps=args.num_max_env_steps
        ),
        backend=pufferlib.vector.Multiprocessing,
        num_envs=args.num_envs,
        num_workers=args.num_envs,  # num_workers == args.num_envs must be true else an error will occur
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs, num_agents) + envs.single_observation_space.shape).to(device) 
    actions = torch.zeros((args.num_steps, args.num_envs, num_agents) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs, num_agents)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs, num_agents)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs, num_agents)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs, num_agents)).to(device)

    if args.model_save_dir is not None:
        model_save_path = os.path.join(args.model_save_dir, args.model_save_filename)
    else:
        model_save_path = None

    model_path = args.model_load_path if args.model_load_path else None
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        agent = torch.load(model_path, map_location=device)
    else:
        print("Initializing new model...")
        # Annoyance: AsyncVectorEnv does not have a driver env
        agent = Agent(envs.driver_env.emulated).to(device)

    if args.run_mode == "train":
        # Training mode: initialize optimizer and training variables
        optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        # Start training loop
        global_step = 0
        start_time = time.time()
        next_obs, _ = envs.reset(seed=args.seed) # returns tensor of shape ([6, 120]), but we need it of shape ([3, 2, 120])
        next_done = torch.zeros((args.num_envs, num_agents)).to(device)

        # Initialize list to keep track of saved models (keeping only the latest 3 models)
        saved_models = []
        last_save_time = time.time()

        if args.ent_coef_decay:
            original_ent_coef = args.ent_coef
            current_ent_coef = args.ent_coef

        # Begin training iteration loop
        for iteration in tqdm(range(1, num_iterations + 1)):
            # Annealing the learning rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            # Step loop for environment interaction
            for step in range(0, args.num_steps):
                # Update global step count and store current observations and dones
                global_step += args.num_envs * num_agents
                obs[step] = torch.Tensor(next_obs.reshape(args.num_envs, num_agents, envs.single_observation_space.shape[0])).to(device) # error here
                dones[step] = torch.Tensor(next_done.reshape(args.num_envs, num_agents)).to(device)

                 # Get actions and values from agentic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(torch.Tensor(next_obs).to(device))
                    values[step] = value.reshape(args.num_envs, num_agents)
                actions[step] = action.reshape(args.num_envs, num_agents)
                logprobs[step] = logprob.reshape(args.num_envs, num_agents)

                 # Execute the game step and collect new observations
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(args.num_envs, num_agents)
                next_obs = torch.as_tensor(next_obs, device=device)
                next_done = torch.as_tensor(next_done, dtype=torch.float32, device=device)
                
                writer.add_scalar(f"charts/mean-reward", rewards[step].mean(), global_step)

                episode_reward_list = []
                episode_length_list = []
                trash_collected_list = []
                for info in infos:               
                    if "final_info" in info:
                        episode_reward_list.append(info["final_info"]["episode_reward"])
                        episode_length_list.append(info["final_info"]["episode_length"])
                        trash_collected_list.append(info["final_info"]["trash_collected"])

                if len(episode_reward_list) > 0:
                    writer.add_scalar("charts/episodic_reward", np.mean(episode_reward_list), global_step)

                if len(episode_length_list) > 0:
                    writer.add_scalar("charts/episodic_length", np.mean(episode_length_list), global_step)

                if len(trash_collected_list) > 0:
                    writer.add_scalar("charts/total_trash_collected", np.mean(trash_collected_list), global_step)


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
            b_inds = np.arange(batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
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
                    if not args.ent_coef_decay:
                        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    else:
                        loss = pg_loss - current_ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if args.ent_coef_decay:
                current_ent_coef -= original_ent_coef / num_iterations

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            if args.ent_coef_decay:
                writer.add_scalar("losses/entropy_coef", current_ent_coef, global_step)
            else:
                writer.add_scalar("losses/entropy_coef", args.ent_coef, global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            # if iteration % 5 == 0:
                # print(f"SPS: {int(global_step / (time.time() - start_time))} | Current Iteration: {iteration} out of {num_iterations}")
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # (Optional) Save model periodically
            # Check if enough time has passed or if we're at the final iteration
            time_since_last_save = time.time() - last_save_time
            if (time_since_last_save >= args.model_save_interval * 60 or iteration == num_iterations) and model_save_path:
                # Define a unique filename for each saved model
                model_filename = f"{model_save_path}_iter_{iteration}.pt"
                torch.save(agent, model_filename)
                last_save_time = time.time()

                # Append the new model to the saved models list
                saved_models.append(model_filename)

                # Remove oldest model if we have more than 3 saved
                if len(saved_models) > 3:
                    oldest_model = saved_models.pop(0)
                    if os.path.exists(oldest_model):
                        os.remove(oldest_model)

    elif args.run_mode == "evaluate":
         # Evaluation mode: run episodes and log performance
        print("Evaluating loaded model")

        # Track results across episodes for final metrics
        episode_reward_list, episode_length_list, trash_collected_list = [], [], []

        # Run the environment for the specified number of evaluation episodes
        for episode in tqdm(range(args.num_eval_episodes), desc='Evaluating Episodes...'):
            obs, _ = envs.reset()
            done = False
            episode_reward = 0

            # Play out one episode, logging rewards and episode information
            for step in range(args.num_max_env_steps):
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
                obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                done = terminations.any() or truncations.any()
                episode_reward += reward.sum().item()

                for info in infos:               
                    if "final_info" in info:
                        episode_reward_list.append(info["final_info"]["episode_reward"])
                        episode_length_list.append(info["final_info"]["episode_length"])
                        trash_collected_list.append(info["final_info"]["trash_remaining"])

        # Summarize evaluation results
        print(f"Average Episode Reward: {np.mean(episode_reward_list)}")
        print(f"Average Episode Length: {np.mean(episode_length_list)}")
        print(f"Average Trash Remaining: {np.mean(trash_collected_list)}")

    elif args.run_mode == "video":
        print("Generating video for loaded model")

        # We are just using Serial vecenv to give a consistent
        # single-agent/multi-agent API for evaluation
        env = pufferlib.vector.make(
            pufferlib.environments.trash_pickup.env_creator(
                grid_size=args.grid_size,
                num_agents=num_agents,
                num_trash=args.num_trash,
                num_bins=args.num_bins,
                max_steps=args.num_max_env_steps
            ),
            env_kwargs={}, 
            backend=pufferlib.vector.Multiprocessing
        )

        agent_model = torch.load(model_path, map_location=device)

        driver = env.driver_env
    
        os.system('clear')
        state = None

        frames = []
        for episode in range(args.num_eval_video_episodes):
            driver.reset()
            ob, info = driver.reset()
            while True:
                render = driver.render()
                frames.append(render)

                with torch.no_grad():
                    agent_observations = [torch.as_tensor(ob[agent_id]).to(device).unsqueeze(0) for agent_id in ob.keys()]
                    ob_convert = torch.cat(agent_observations, dim=0)  # Batch dimension for all agents

                    # Get actions for each agent in batch
                    actions, _, _, _ = agent_model.get_action_and_value(ob_convert)
                    
                    # Convert actions back to numpy and reshape for each agent
                    actions = actions.cpu().numpy()
                    action_dict = {f"agent_{i}": actions[i] for i in range(num_agents)}

                ob, _, dones, _, infos = driver.step(action_dict)
                
                if dones['agent_0']: # all dones
                    break

        # Save frames as gif
        print("Frames collected, creating video...")
        print("Warning, script may hang even after video is created, check if video is created and if so then manually terminate program.")
        import imageio
        imageio.mimsave('examples/trash_pickup_vid.gif', frames, fps=3, loop=0)

        # NOTE: If you get this error - "TypeError: write_frames() got an unexpected keyword argument 'audio_path'""
        # Try the following command and then rerun: pip install --upgrade imageio imageio-ffmpeg
    else:
        print(f"Unhandled run_mode of: {args.run_mode}")

    envs.close()
    try:
        writer.close()
    except:
        pass

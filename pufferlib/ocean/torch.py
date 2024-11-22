from torch import nn
import torch
import torch.nn.functional as F

from functools import partial
import pufferlib.models

from pufferlib.models import Default as Policy
Recurrent = pufferlib.models.LSTMWrapper

class Snake(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128, **kwargs):
        super().__init__()
        self.network= nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(8, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(cnn_channels, hidden_size)),
            nn.ReLU(),
        )
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        observations = F.one_hot(observations.long(), 8).permute(0, 3, 1, 2).float()
        return self.network(observations), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

class Grid(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128, **kwargs):
        super().__init__()
        self.cnn = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(7, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.Flatten(),
        )
        self.flat = pufferlib.pytorch.layer_init(nn.Linear(3, 32))
        self.proj = pufferlib.pytorch.layer_init(nn.Linear(32+cnn_channels, hidden_size))

        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.decoder_logstd = nn.Parameter(torch.zeros(
                1, env.single_action_space.shape[0]))
        else:
            self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, 6), std=0.01)

        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        cnn_features = observations[:, :-3].view(-1, 11, 11).long()
        cnn_features = F.one_hot(cnn_features, 7).permute(0, 3, 1, 2).float()
        cnn_features = self.cnn(cnn_features)

        flat_features = observations[:, -3:].float() / 255.0
        flat_features = self.flat(flat_features)

        features = torch.cat([cnn_features, flat_features], dim=1)
        features = F.relu(self.proj(F.relu(features)))
        return features, None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        value = self.value_fn(flat_hidden)
        if self.is_continuous:
            mean = self.decoder_mean(flat_hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            probs = torch.distributions.Normal(mean, std)
            batch = flat_hidden.shape[0]
            return probs, value
        else:
            action = self.actor(flat_hidden).split(3, dim=1)
            return action, value

class MOBA(nn.Module):
    def __init__(self, env, cnn_channels=128, hidden_size=128, **kwargs):
        super().__init__()
        self.cnn = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(16 + 3, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.Flatten(),
        )
        self.flat = pufferlib.pytorch.layer_init(nn.Linear(26, 128))
        self.proj = pufferlib.pytorch.layer_init(nn.Linear(128+cnn_channels, hidden_size))

        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.decoder_logstd = nn.Parameter(torch.zeros(
                1, env.single_action_space.shape[0]))
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()
            self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)

        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        cnn_features = observations[:, :-26].view(-1, 11, 11, 4).long()
        if cnn_features[:, :, :, 0].max() > 15:
            print('Invalid map value:', cnn_features[:, :, :, 0].max())
            breakpoint()
            exit(1)
        map_features = F.one_hot(cnn_features[:, :, :, 0], 16).permute(0, 3, 1, 2).float()
        extra_map_features = (cnn_features[:, :, :, -3:].float() / 255).permute(0, 3, 1, 2)
        cnn_features = torch.cat([map_features, extra_map_features], dim=1)
        #print('observations 2d: ', map_features[0].cpu().numpy().tolist())
        cnn_features = self.cnn(cnn_features)
        #print('cnn features: ', cnn_features[0].detach().cpu().numpy().tolist())

        flat_features = observations[:, -26:].float() / 255.0
        #print('observations 1d: ', flat_features[0, 0])
        flat_features = self.flat(flat_features)
        #print('flat features: ', flat_features[0].detach().cpu().numpy().tolist())

        features = torch.cat([cnn_features, flat_features], dim=1)
        features = F.relu(self.proj(F.relu(features)))
        #print('features: ', features[0].detach().cpu().numpy().tolist())
        return features, None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        #print('lstm: ', flat_hidden[0].detach().cpu().numpy().tolist())
        value = self.value_fn(flat_hidden)
        if self.is_continuous:
            mean = self.decoder_mean(flat_hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            probs = torch.distributions.Normal(mean, std)
            batch = flat_hidden.shape[0]
            return probs, value
        else:
            action = self.actor(flat_hidden)
            action = torch.split(action, self.atn_dim, dim=1)

            #argmax_samples = [torch.argmax(a, dim=1).detach().cpu().numpy().tolist() for a in action]
            #print('argmax samples: ', argmax_samples)

            return action, value


class TrashPickup(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.num_trash = env.num_trash
        self.num_bins = env.num_bins
        self.num_agents_per_env = env.num_agents_per_env

        self.trash_net = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(3 * env.num_trash, 32)),  # (presence, x, y) for each trash
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(32, 16)),
            nn.ReLU(),
        )

        self.bin_net = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(2 * env.num_bins, 8)),  # (x, y) for each bin
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(8, 4)),
            nn.ReLU(),
        )

        self.other_agent_net = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(3 * (env.num_agents_per_env - 1), 16)),  # (x, y, carrying) for each other agent
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(16, 8)),
            nn.ReLU(),
        )

        self.position_net = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(2, 8)),
            nn.ReLU(),
        )
        
        self.carrying_net = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(1, 2)),
            nn.ReLU(),
        )

        self.proj = nn.Sequential( 
            nn.Linear(16 + 4 + 8 + 8 + 2, 32),  # Combined features size
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(16, 8)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(8, 4), std=0.01),  # 4 actions
        )

        self.critic = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(16, 8)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(8, 1), std=1),  # Value prediction
        )

        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

    def forward(self, observations):
        hidden, _ = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def encode_observations(self, observations):
        """
        Encode observations for each agent.
        """

        features = []
        obs_index = 0

        # Agent's own position and carrying status
        agent_position = observations[:, obs_index : obs_index + 2]
        carrying_status = observations[:, obs_index + 2 : obs_index + 3]
        obs_index += 3

        # Other agents
        other_agents = observations[:, obs_index : obs_index + 3 * (self.num_agents_per_env - 1)]
        obs_index += 3 * (self.num_agents_per_env - 1)

        # Trash data
        trash_data = observations[:, obs_index : obs_index + 3 * self.num_trash]
        obs_index += 3 * self.num_trash

        # Bin data
        bin_data = observations[:, obs_index : obs_index + 2 * self.num_bins]

        # Pass through sub-networks
        trash_features = self.trash_net(trash_data)
        bin_features = self.bin_net(bin_data)
        other_agent_features = self.other_agent_net(other_agents)
        position_features = self.position_net(agent_position)
        carrying_features = self.carrying_net(carrying_status)

        # Combine features
        concat = torch.cat(
            [trash_features, bin_features, other_agent_features, position_features, carrying_features],
            dim=1,
        )
        features.append(self.proj(concat))

        return torch.stack(features), None

    def decode_actions(self, hidden):
        """
        Decode actions and values from the hidden state.
        """
        value = self.critic(hidden)
        if self.is_continuous:
            mean = self.actor(hidden)
            logstd = torch.zeros_like(mean)
            std = torch.exp(logstd)
            probs = torch.distributions.Normal(mean, std)
            return probs, value
        else:
            logits = self.actor(hidden)
            logits = logits.view(-1, 4)  # Ensure correct shape for sampling
            return logits, value

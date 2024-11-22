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
    def __init__(self, emulated):
        super().__init__()
        self.dtype = pufferlib.pytorch.nativize_dtype(emulated)

        def get_conv_output_size(input_shape):
            """Calculates the size of the output from the convolution layers."""
            dummy_input = torch.zeros(1, *input_shape)  # Batch size of 1 with given input shape
            output = self.conv2(self.conv1(dummy_input))  # Pass through conv layers
            return int(np.prod(output.shape[1:]))  # Flatten dimensions, except batch
        
        # Define network for processing the grid with multiple channels (4 channels for 0, 1, 2, 3)
        input_channels = 4
        
        # Define the convolutional layers
        self.conv1 = layer_init(nn.Conv2d(input_channels, 32, 3, stride=1))
        self.conv2 = layer_init(nn.Conv2d(32, 64, 3, stride=1))

        # Calculate the output size of the conv layers
        conv_output_size = get_conv_output_size((input_channels, args.grid_size, args.grid_size))

        
        self.grid_net = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(conv_output_size, 32)),
            nn.ReLU(),
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
        self.proj = nn.Linear(32 + 32 + 32, 256)  # Adjusted projection layer size based on concatenated features
        self.actor = layer_init(nn.Linear(256, 4), std=0.01)  # 4 represents the number of possible actions
        self.critic = layer_init(nn.Linear(256, 1), std=1)  # 1 represents the value of the critic network (or sometimes called value network) 

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
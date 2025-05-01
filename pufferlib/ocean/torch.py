from typing import Any, Tuple

from gymnasium import spaces

from torch import nn
import torch
from torch.distributions.normal import Normal
from torch import nn
import torch.nn.functional as F

import pufferlib
import pufferlib.models

from pufferlib.models import Default as Policy
from pufferlib.models import Convolutional as Conv
Recurrent = pufferlib.models.LSTMWrapper
from pufferlib.pytorch import layer_init, _nativize_dtype, nativize_tensor
import numpy as np


class Snake(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128,
            use_p3o=False, p3o_horizon=32, use_diayn=False, diayn_skills=8, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        self.use_diayn = use_diayn

        encode_dim = cnn_channels
        if use_diayn:
            encode_dim += diayn_skills
            self.diayn_skills = diayn_skills
            '''
            self.diayn_discriminator = nn.Sequential(
                pufferlib.pytorch.layer_init(
                    nn.Conv2d(64, cnn_channels, 5, stride=3)),
                nn.ReLU(),
                pufferlib.pytorch.layer_init(
                    nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                pufferlib.pytorch.layer_init(nn.Linear(cnn_channels, diayn_skills)),
            )
            '''
            self.diayn_discriminator = nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(64*env.single_action_space.n, hidden_size)),
                nn.ReLU(),
                pufferlib.pytorch.layer_init(nn.Linear(hidden_size, diayn_skills)),
            )
 
        self.network= nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(8, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(encode_dim, hidden_size)),
            nn.ReLU(),
        )
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)

        self.use_p3o = use_p3o
        self.p3o_horizon = p3o_horizon
        if use_p3o:
            self.value_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, p3o_horizon), std=1)
            self.value_logstd = nn.Parameter(torch.zeros(1, p3o_horizon))
        else:
            self.value = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, 1), std=1)

    def discrim_forward(self, obs):
        #obs = F.one_hot(obs.long(), 8).permute(0, 1, 4, 2, 3).float()
        #B, f, c, h, w = obs.shape
        #obs = obs.reshape(B, f*c, h, w)
        return self.diayn_discriminator(obs)

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return (actions, value), hidden

    def encode_observations(self, observations, state=None):
        observations = F.one_hot(observations.long(), 8).permute(0, 3, 1, 2).float()
        hidden = self.network(observations)

        if self.use_diayn:
            z_one_hot = F.one_hot(state.diayn_z, self.diayn_skills).float()
            hidden = torch.cat([hidden, z_one_hot], dim=1)

        return self.proj(hidden)

    def decode_actions(self, hidden):
        action = self.actor(hidden)

        if self.use_p3o:
            value_mean = self.value_mean(hidden)
            value_logstd = self.value_logstd.expand_as(value_mean)
            return action, value_mean, value_logstd
        else:
            value = self.value(hidden)
            return action, value


class Boids(nn.Module):
    def __init__(self, env, hidden_size=128, use_p3o=False, p3o_horizon=32, **kwargs):
        super().__init__()
        self.num_boids = env.single_observation_space.shape[0]
        self.hidden_size = hidden_size
        self.is_continuous = True
        self.use_p3o = use_p3o
        self.p3o_horizon = p3o_horizon
        
        # Process each boid separately with a shared network
        self.boid_encoder = nn.Sequential(
            layer_init(nn.Linear(4, hidden_size)),  # Each boid has 4 observation values (x, y, vx, vy)
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )
        
        # Process aggregated boid features
        self.global_encoder = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )
        
        # Output mean and log_std for each boid's action (vx, vy)
        self.action_mean = layer_init(nn.Linear(hidden_size, 2), std=0.01)
        self.action_logstd = nn.Parameter(torch.zeros(1, 2))
        
        if use_p3o:
            self.value_mean = layer_init(nn.Linear(hidden_size, p3o_horizon), std=1)
            self.value_logstd = nn.Parameter(torch.zeros(1, p3o_horizon))
        else:
            self.value = layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        batch_size = observations.shape[0]
        
        # Reshape to process each boid separately
        # [batch_size, num_boids, 4] -> [batch_size * num_boids, 4]
        obs_flat = observations.reshape(-1, 4)
        
        # Process each boid
        boid_features = self.boid_encoder(obs_flat)
        
        # Reshape back and average across boids
        # [batch_size * num_boids, hidden_size] -> [batch_size, num_boids, hidden_size]
        boid_features = boid_features.reshape(batch_size, self.num_boids, self.hidden_size)
        
        # Get a global feature for all boids
        global_features = torch.mean(boid_features, dim=1)
        global_features = self.global_encoder(global_features)
        
        # Expand global features to apply to each boid
        # [batch_size, hidden_size] -> [batch_size, num_boids, hidden_size]
        global_features_expanded = global_features.unsqueeze(1).expand(-1, self.num_boids, -1)
        
        # Reshape for action prediction
        # [batch_size, num_boids, hidden_size] -> [batch_size * num_boids, hidden_size]
        action_features = global_features_expanded.reshape(-1, self.hidden_size)
        
        # Predict action mean and std for each boid
        action_mean = self.action_mean(action_features)
        action_logstd = self.action_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        # Create normal distribution
        dist = Normal(action_mean, action_std)
        
        # Sample actions
        if self.training:
            actions = dist.sample()
        else:
            actions = action_mean
            
        # Calculate log probabilities
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        # Reshape actions back to [batch_size, num_boids, 2]
        actions = actions.reshape(batch_size, self.num_boids, 2)
        
        # Compute value
        if self.use_p3o:
            value_mean = self.value_mean(global_features)
            value_logstd = self.value_logstd.expand_as(value_mean)
            value = pufferlib.namespace(mean=value_mean, std=torch.exp(value_logstd))
        else:
            value = self.value(global_features)
            
        # Reshape log_probs and entropy
        log_probs = log_probs.reshape(batch_size, self.num_boids).mean(dim=1)
        entropy = entropy.reshape(batch_size, self.num_boids).mean(dim=1)
        
        return actions, log_probs, entropy, value


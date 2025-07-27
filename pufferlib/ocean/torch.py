from types import SimpleNamespace
from typing import Any, Tuple

from gymnasium import spaces

from torch import nn
import torch
from torch.distributions.normal import Normal
from torch import nn
import torch.nn.functional as F

import pufferlib
import pufferlib.models

from pufferlib.models import LSTMWrapper as Recurrent
from pufferlib.models import Default as _OldDefault

from pufferlib.pytorch import layer_init, _nativize_dtype, nativize_tensor
import numpy as np


class Boids(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(4, hidden_size)),
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
        )
        self.action_vec = tuple(env.single_action_space.nvec)
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, sum(self.action_vec)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        batch, n, = observations.shape
        return self.network(observations.reshape(batch, n//4, 4)).max(dim=1)[0]

    def decode_actions(self, flat_hidden, state=None):
        value = self.value_fn(flat_hidden)
        action = self.actor(flat_hidden).split(self.action_vec, dim=1)
        return action, value

class NMMO3LSTM(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512):
        super().__init__(env, policy, input_size, hidden_size)

class NMMO3(nn.Module):
    def __init__(self, env, hidden_size=512, output_size=512, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        #self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.num_actions = env.single_action_space.n
        self.factors = np.array([4, 4, 17, 5, 3, 5, 5, 5, 7, 4])
        offsets = torch.tensor([0] + list(np.cumsum(self.factors)[:-1])).view(1, -1, 1, 1)
        self.register_buffer('offsets', offsets)
        self.cum_facs = np.cumsum(self.factors)

        self.multihot_dim = self.factors.sum()
        self.is_continuous = False

        self.map_2d = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(self.multihot_dim, 128, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(128, 128, 3, stride=1)),
            nn.Flatten(),
        )

        self.player_discrete_encoder = nn.Sequential(
            nn.Embedding(128, 32),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(1817, hidden_size)),
            nn.ReLU(),
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(output_size, self.num_actions), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)

    def forward(self, x, state=None):
        hidden = self.encode_observations(x)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        batch = observations.shape[0]
        ob_map = observations[:, :11*15*10].view(batch, 11, 15, 10)
        ob_player = observations[:, 11*15*10:-10]
        ob_reward = observations[:, -10:]

        batch = ob_map.shape[0]
        map_buf = torch.zeros(batch, 59, 11, 15, dtype=torch.float32, device=observations.device)
        codes = ob_map.permute(0, 3, 1, 2) + self.offsets
        map_buf.scatter_(1, codes, 1)
        ob_map = self.map_2d(map_buf)

        player_discrete = self.player_discrete_encoder(ob_player.int())

        obs = torch.cat([ob_map, player_discrete, ob_player.to(ob_map.dtype), ob_reward], dim=1)
        obs = self.proj(obs)
        return obs

    def decode_actions(self, flat_hidden):
        flat_hidden = self.layer_norm(flat_hidden)
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

class Terraform(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        self.local_net_2d = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(2, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.global_net_2d = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(2, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.net_1d = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(5, hidden_size)),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size + cnn_channels*5, hidden_size)),
            nn.ReLU(),
        )
        self.atn_dim = env.single_action_space.nvec.tolist()
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations, state)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        # breakpoint()
        obs_2d = observations[:, :242].reshape(-1, 2, 11, 11).float()
        obs_1d = observations[:, 242:247].reshape(-1, 5).float()
        location_2d = observations[:, 247:].reshape(-1,2, 6, 6).float()
        hidden_local_2d = self.local_net_2d(obs_2d)
        hidden_global_2d = self.global_net_2d(location_2d)
        hidden_1d = self.net_1d(obs_1d)
        hidden = torch.cat([hidden_local_2d, hidden_global_2d, hidden_1d], dim=1)
        return self.proj(hidden)

    def decode_actions(self, hidden):
        action = self.actor(hidden)
        action = torch.split(action, self.atn_dim, dim=1)
        #action = [head(hidden) for head in self.actor]
        value = self.value(hidden)
        return action, value

class G2048(nn.Module):
    def __init__(self, env, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        self.encoder= torch.nn.Sequential(
            nn.Linear(12*np.prod(env.single_observation_space.shape), hidden_size),
            nn.GELU(),
        )
        self.decoder = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward(self, x, state=None):
        return self.forward_eval(x, state)

    def encode_observations(self, observations, state=None):
        observations = F.one_hot(observations.long(), 12).view(-1, 12*16).float()
        #observations = observations.float().view(-1, 16) / 11.0
        return self.encoder(observations)

    def decode_actions(self, hidden):
        action = self.decoder(hidden)
        value = self.value(hidden)
        return action, value

class Snake(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        encode_dim = cnn_channels

        '''
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
 
        '''
        self.encoder= torch.nn.Sequential(
            nn.Linear(8*np.prod(env.single_observation_space.shape), hidden_size),
            nn.GELU(),
        )
        self.decoder = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        #observations = F.one_hot(observations.long(), 8).permute(0, 3, 1, 2).float()
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        observations = F.one_hot(observations.long(), 8).view(-1, 11*11*8).float()
        return self.encoder(observations)

    def decode_actions(self, hidden):
        action = self.decoder(hidden)
        value = self.value(hidden)
        return action, value

'''
class Snake(pufferlib.models.Default):
    def __init__(self, env, hidden_size=128):
        super().__init__()

    def encode_observations(self, observations, state=None):
        observations = F.one_hot(observations.long(), 8).view(-1, 11*11*8).float()
        super().encode_observations(observations, state)
'''

class Grid(nn.Module):
    def __init__(self, env, cnn_channels=32, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(32, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.Flatten(),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(cnn_channels, hidden_size)),
            nn.ReLU(),
        )

        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.decoder_logstd = nn.Parameter(torch.zeros(
                1, env.single_action_space.shape[0]))
        else:
            num_actions = env.single_action_space.n
            self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, num_actions), std=0.01)

        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        hidden = observations.view(-1, 11, 11).long()
        hidden = F.one_hot(hidden, 32).permute(0, 3, 1, 2).float()
        hidden = self.network(hidden)
        return hidden

    def decode_actions(self, flat_hidden, state=None):
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
            return action, value

class Go(nn.Module):
    def __init__(self, env, cnn_channels=64, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        # 3 categories 2 boards. 
        # categories = player, opponent, empty
        # boards = current, previous
        self.cnn = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(2, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride = 1)),
            nn.Flatten(),
        )

        obs_size = env.single_observation_space.shape[0]
        self.grid_size = int(np.sqrt((obs_size-2)/2))
        output_size = self.grid_size - 4
        cnn_flat_size = cnn_channels * output_size * output_size
        
        self.flat = pufferlib.pytorch.layer_init(nn.Linear(2,32))
        
        self.proj = pufferlib.pytorch.layer_init(nn.Linear(cnn_flat_size + 32, hidden_size))

        self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.n), std=0.01)

        self.value_fn = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, 1), std=1)
   
    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        grid_size = int(np.sqrt((observations.shape[1] - 2) / 2))
        full_board = grid_size * grid_size 
        black_board = observations[:, :full_board].view(-1,1, grid_size,grid_size).float()
        white_board = observations[:, full_board:-2].view(-1,1, grid_size, grid_size).float()
        board_features = torch.cat([black_board, white_board],dim=1)
        flat_feature1 = observations[:, -2].unsqueeze(1).float()
        flat_feature2 = observations[:, -1].unsqueeze(1).float()
        # Pass board through cnn
        cnn_features = self.cnn(board_features)
        # Pass extra feature
        flat_features = torch.cat([flat_feature1, flat_feature2],dim=1)
        flat_features = self.flat(flat_features)
        # pass all features
        features = torch.cat([cnn_features, flat_features], dim=1)
        features = F.relu(self.proj(features))

        return features

    def decode_actions(self, flat_hidden, state=None):
        value = self.value_fn(flat_hidden)
        action = self.actor(flat_hidden)
        return action, value
    
class MOBA(nn.Module):
    def __init__(self, env, cnn_channels=128, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
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

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        cnn_features = observations[:, :-26].view(-1, 11, 11, 4).long()
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
        return features

    def decode_actions(self, flat_hidden):
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
    def __init__(self, env, cnn_channels=32, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        self.network= nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(5, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(cnn_channels, hidden_size)),
        )
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        observations = observations.view(-1, 5, 11, 11).float()
        return self.network(observations)

    def decode_actions(self, flat_hidden):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

class TowerClimbLSTM(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size = 256, hidden_size = 256):
        super().__init__(env, policy, input_size, hidden_size)

class TowerClimb(nn.Module):
    def __init__(self, env, cnn_channels=16, hidden_size = 256, **kwargs):
        self.hidden_size = hidden_size
        self.is_continuous = False
        super().__init__()
        self.network = nn.Sequential(
                pufferlib.pytorch.layer_init(
                    nn.Conv3d(1, cnn_channels, 3, stride = 1)),
                nn.ReLU(),
                pufferlib.pytorch.layer_init(
                    nn.Conv3d(cnn_channels, cnn_channels, 3, stride=1)),
                nn.Flatten()       
        )
        cnn_flat_size = cnn_channels * 1 * 1 * 5

        # Process player obs
        self.flat = pufferlib.pytorch.layer_init(nn.Linear(3,16))

        # combine
        self.proj = pufferlib.pytorch.layer_init(
                nn.Linear(cnn_flat_size + 16, hidden_size))
        self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.n), std = 0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, 1 ), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        board_state = observations[:,:225]
        player_info = observations[:, -3:] 
        board_features = board_state.view(-1, 1, 5,5,9).float()
        cnn_features = self.network(board_features)
        flat_features = self.flat(player_info.float())
        
        features = torch.cat([cnn_features,flat_features],dim = 1)
        features = self.proj(features)
        return features
    
    def decode_actions(self, flat_hidden):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        
        return action, value


class ImpulseWarsLSTM(Recurrent):
    def __init__(self, env: pufferlib.PufferEnv, policy: nn.Module, input_size: int = 512, hidden_size: int = 512):
        super().__init__(env, policy, input_size, hidden_size)


class ImpulseWarsPolicy(nn.Module):
    def __init__(
        self,
        env: pufferlib.PufferEnv,
        cnn_channels: int = 64,
        weapon_type_embedding_dims: int = 2,
        input_size: int = 512,
        hidden_size: int = 512,
        batch_size: int = 131_072,
        num_drones: int = 2,
        continuous: bool = False,
        is_training: bool = True,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.is_continuous = continuous

        self.numDrones = num_drones
        self.isTraining = is_training
        from pufferlib.ocean.impulse_wars import binding
        self.obsInfo = SimpleNamespace(**binding.get_consts(self.numDrones))

        self.discreteFactors = np.array(
            [self.obsInfo.wallTypes] * self.obsInfo.numNearWallObs
            + [self.obsInfo.wallTypes + 1] * self.obsInfo.numFloatingWallObs
            + [self.numDrones + 1] * self.obsInfo.numProjectileObs,
        )
        discreteOffsets = torch.tensor([0] + list(np.cumsum(self.discreteFactors)[:-1]), device=device).view(
            1, -1
        )
        self.register_buffer("discreteOffsets", discreteOffsets, persistent=False)
        self.discreteMultihotDim = self.discreteFactors.sum()

        multihotBuffer = torch.zeros(batch_size, self.discreteMultihotDim, device=device)
        self.register_buffer("multihotOutput", multihotBuffer, persistent=False)

        # most of the observation is a 2D array of bytes, but the end
        # contains around 200 floats; this allows us to treat the end
        # of the observation as a float array
        _, *self.dtype = _nativize_dtype(
            np.dtype((np.uint8, (self.obsInfo.continuousObsBytes,))),
            np.dtype((np.float32, (self.obsInfo.continuousObsSize,))),
        )
        self.dtype = tuple(self.dtype)

        self.weaponTypeEmbedding = nn.Embedding(self.obsInfo.weaponTypes, weapon_type_embedding_dims)

        # each byte in the map observation contains 4 values:
        # - 2 bits for wall type
        # - 1 bit for is floating wall
        # - 1 bit for is weapon pickup
        # - 3 bits for drone index
        self.register_buffer(
            "unpackMask",
            torch.tensor([0x60, 0x10, 0x08, 0x07], dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer("unpackShift", torch.tensor([5, 4, 3, 0], dtype=torch.uint8), persistent=False)

        self.mapObsInputChannels = (self.obsInfo.wallTypes + 1) + 1 + 1 + self.numDrones
        self.mapCNN = nn.Sequential(
            layer_init(
                nn.Conv2d(
                    self.mapObsInputChannels,
                    cnn_channels,
                    kernel_size=5,
                    stride=3,
                )
            ),
            nn.ReLU(),
            layer_init(nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        cnnOutputSize = self._computeCNNShape()

        featuresSize = (
            cnnOutputSize
            + (self.obsInfo.numNearWallObs * (self.obsInfo.wallTypes + self.obsInfo.nearWallPosObsSize))
            + (
                self.obsInfo.numFloatingWallObs
                * (self.obsInfo.wallTypes + 1 + self.obsInfo.floatingWallInfoObsSize)
            )
            + (
                self.obsInfo.numWeaponPickupObs
                * (weapon_type_embedding_dims + self.obsInfo.weaponPickupPosObsSize)
            )
            + (
                self.obsInfo.numProjectileObs
                * (weapon_type_embedding_dims + self.obsInfo.projectileInfoObsSize + self.numDrones + 1)
            )
            + ((self.numDrones - 1) * (weapon_type_embedding_dims + self.obsInfo.enemyDroneObsSize))
            + (self.obsInfo.droneObsSize + weapon_type_embedding_dims)
            + self.obsInfo.miscObsSize
        )

        self.encoder = nn.Sequential(
            layer_init(nn.Linear(featuresSize, input_size)),
            nn.ReLU(),
        )

        if self.is_continuous:
            self.actorMean = layer_init(nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.actorLogStd = nn.Parameter(torch.zeros(1, env.single_action_space.shape[0]))
        else:
            self.actionDim = env.single_action_space.nvec.tolist()
            self.actor = layer_init(nn.Linear(hidden_size, sum(self.actionDim)), std=0.01)

        self.critic = layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def forward(self, obs: torch.Tensor, state = None) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_observations(obs)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def unpack(self, batchSize: int, obs: torch.Tensor) -> torch.Tensor:
        # prepare map obs to be unpacked
        mapObs = obs[:, : self.obsInfo.mapObsSize].reshape((batchSize, -1, 1))
        # unpack wall types, weapon pickup types, and drone indexes
        mapObs = (mapObs & self.unpackMask) >> self.unpackShift
        # reshape so channels are first, required for torch conv2d
        return mapObs.permute(0, 2, 1).reshape(
            (batchSize, 4, self.obsInfo.mapObsRows, self.obsInfo.mapObsColumns)
        )

    def encode_observations(self, obs: torch.Tensor, state: Any = None) -> torch.Tensor:
        batchSize = obs.shape[0]

        mapObs = self.unpack(batchSize, obs)

        # one hot encode wall types
        wallTypeObs = mapObs[:, 0, :, :].long()
        wallTypes = F.one_hot(wallTypeObs, self.obsInfo.wallTypes + 1).permute(0, 3, 1, 2).float()

        # unsqueeze floating wall booleans (is wall a floating wall)
        floatingWallObs = mapObs[:, 1, :, :].unsqueeze(1)

        # unsqueeze map pickup booleans (does map tile contain a weapon pickup)
        mapPickupObs = mapObs[:, 2, :, :].unsqueeze(1)

        # one hot drone indexes
        droneIndexObs = mapObs[:, 3, :, :].long()
        droneIndexes = F.one_hot(droneIndexObs, self.numDrones).permute(0, 3, 1, 2).float()

        # combine all map observations and feed through CNN
        mapObs = torch.cat((wallTypes, floatingWallObs, mapPickupObs, droneIndexes), dim=1)
        map = self.mapCNN(mapObs)

        # process discrete observations
        multihotInput = (
            obs[:, self.obsInfo.nearWallTypesObsOffset : self.obsInfo.projectileTypesObsOffset]
            + self.discreteOffsets
        )
        multihotOutput = self.multihotOutput[:batchSize].zero_()
        multihotOutput.scatter_(1, multihotInput.long(), 1)

        weaponTypeObs = obs[:, self.obsInfo.projectileTypesObsOffset : self.obsInfo.discreteObsSize].int()
        weaponTypes = self.weaponTypeEmbedding(weaponTypeObs).float()
        weaponTypes = torch.flatten(weaponTypes, start_dim=1, end_dim=-1)

        # process continuous observations
        continuousObs = nativize_tensor(obs[:, self.obsInfo.continuousObsOffset :], self.dtype)
        # combine all observations and feed through final linear encoder
        features = torch.cat((map, multihotOutput, weaponTypes, continuousObs), dim=-1)

        return self.encoder(features)

    def decode_actions(self, hidden: torch.Tensor):
        if self.is_continuous:
            actionMean = self.actorMean(hidden)
            if self.isTraining:
                actionLogStd = self.actorLogStd.expand_as(actionMean)
                actionStd = torch.exp(actionLogStd)
                action = Normal(actionMean, actionStd)
            else:
                action = actionMean
        else:
            action = self.actor(hidden)
            action = torch.split(action, self.actionDim, dim=1)

        value = self.critic(hidden)

        return action, value

    def _computeCNNShape(self) -> int:
        mapSpace = spaces.Box(
            low=0,
            high=1,
            shape=(self.mapObsInputChannels, self.obsInfo.mapObsRows, self.obsInfo.mapObsColumns),
            dtype=np.float32,
        )

        with torch.no_grad():
            t = torch.as_tensor(mapSpace.sample()[None])
            return self.mapCNN(t).shape[1]

class GPUDrive(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(6, input_size)),
            # nn.ReLU(),
            # pufferlib.pytorch.layer_init(
            #    nn.Linear(input_size, input_size))
        )
        max_road_objects = 13
        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(max_road_objects, input_size)),
            # nn.ReLU(),
            # pufferlib.pytorch.layer_init(
            #    nn.Linear(input_size, input_size))
        )
        max_partner_objects = 7
        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(max_partner_objects, input_size)),
            # nn.ReLU(),
            # pufferlib.pytorch.layer_init(
            #    nn.Linear(input_size, input_size))
        )

        '''
        self.post_mask_road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(input_size, input_size)),
        )
        self.post_mask_partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(input_size, input_size)),
        )
        '''
        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(3*input_size,  hidden_size)),
        )
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

        self.atn_dim = env.single_action_space.nvec.tolist()
        self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, sum(self.atn_dim)), std = 0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, 1 ), std=1)
    
    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)
   
    def encode_observations(self, observations, state=None):
        ego_dim = 6
        partner_dim = 63 * 7
        road_dim = 200*7
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim:ego_dim+partner_dim]
        road_obs = observations[:, ego_dim+partner_dim:ego_dim+partner_dim+road_dim]
        
        partner_objects = partner_obs.view(-1, 63, 7)
        road_objects = road_obs.view(-1, 200, 7)
        road_continuous = road_objects[:, :, :6]  # First 6 features
        road_categorical = road_objects[:, :, 6]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)  # Shape: [batch, 200, 7]
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)

        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)
        
        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)
        
        # Pass through shared embedding
        embedding = F.relu(self.shared_embedding(concat_features))
        # embedding = self.shared_embedding(concat_features)
        return embedding
    
    def decode_actions(self, flat_hidden):
        action = self.actor(flat_hidden)
        action = torch.split(action, self.atn_dim, dim=1)
        value = self.value_fn(flat_hidden)
        return action, value

# class ChessLSTM(Recurrent):
#     def __init__(self, env, policy, input_size, hidden_size):
#         super().__init__(env, policy, input_size, hidden_size)

# class ChessRecurrent(nn.Module):
#     """Optimized MLP encoder/decoder for chess observations with UCI action space."""
#     def __init__(self, env=None, hidden_size=256, num_actions=1968, **kwargs):
#         super().__init__()
#         # Simple MLP encoder for flattened chess board (much faster than CNN)
#         self.board_encoder = nn.Sequential(
#             nn.Linear(1472, 512),  # 23 * 8 * 8 = 1472
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU()
#         )
#         # Combiner projects board_features to hidden state
#         self.combiner = nn.Sequential(
#             nn.Linear(256, hidden_size),
#             nn.ReLU()
#         )
#         # Output heads for UCI action space
#         self.policy_head = nn.Linear(hidden_size, num_actions)
#         self.value_head = nn.Sequential(
#             nn.Linear(hidden_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )
#         self.is_continuous = False
#         self.is_multidiscrete = False
#         self.action_size = num_actions

#     def encode_observations(self, obs, state=None):
#         # For LSTM compatibility, we only encode board features here
#         # Mask will be handled in the LSTM wrapper's forward method
#         board_state = obs[:, :1472]  # 23 channels * 8 * 8 = 1472 (flattened)
#         board_features = self.board_encoder(board_state)
#         hidden = self.combiner(board_features)
#         return hidden

#     def get_action(self, obs, temperature=1.0):
#         """Get action with proper UCI-based legal action masking"""
        
#         # 1. Get legal actions from the legal mask in observations
#         legal_mask = obs[:, 1472:3440]  # 1968 UCI legal move mask
#         legal_actions = torch.where(legal_mask[0] > 0.5)[0]
        
#         if len(legal_actions) == 0:
#             # No legal moves - should not happen in normal chess
#             print("WARNING: No legal actions available!")
#             return 0
        
#         # 2. Create action mask (CRITICAL!)
#         action_mask = torch.zeros(self.action_size, device=obs.device)
#         action_mask[legal_actions] = 1.0
        
#         # 3. Get network output (logits)
#         board_state = obs[:, :1344]
#         board_features = self.board_encoder(board_state)
#         hidden = self.combiner(board_features)
#         logits = self.policy_head(hidden)
        
#         # 4. Apply mask to PREVENT illegal actions
#         # Method 1: Set illegal action logits to -infinity
#         masked_logits = logits.clone()
#         masked_logits[0, action_mask == 0] = float('-inf')
        
#         # 5. Sample action (now guaranteed to be legal)
#         probs = F.softmax(masked_logits / temperature, dim=-1)
#         action = torch.multinomial(probs, 1).item()
        
#         return action

#     # File: pufferlib/ocean/torch.py
#     # Inside the ChessRecurrent class

#     def decode_actions(self, hidden, legal_mask):
#         # Get raw logits for UCI actions
#         raw_logits = self.policy_head(hidden)
#         print(f"[POLICY DIAGNOSTIC] decode_actions called.")

#         # --- START OF NEW DIAGNOSTIC ---
#         # Add a check that only runs once to avoid spamming the log
#         if not hasattr(self, '_mask_printed'):
#             num_legal_moves = torch.sum(legal_mask).item()
#             print(f"[POLICY DIAGNOSTIC] decode_actions received legal_mask.")
#             print(f"[POLICY DIAGNOSTIC] Mask shape: {legal_mask.shape}, Num legal moves in mask: {num_legal_moves}")
#             # Print a small sample of the mask
#             print(f"[POLICY DIAGNOSTIC] Mask sample (first 10): {legal_mask[0, :10]}")
#             self._mask_printed = True
#         # --- END OF NEW DIAGNOSTIC ---

#         # Optimized vectorized masking
#         masked_logits = raw_logits.masked_fill(legal_mask < 0.5, -1e8)

#         value = self.value_head(hidden).squeeze(-1)

#         return masked_logits, value

#     # def decode_actions(self, hidden, legal_mask):
#     #     # Get raw logits for UCI actions
#     #     raw_logits = self.policy_head(hidden)
        
#     #     # Optimized vectorized masking: convert legal_mask (0/1) to -inf/0 mask
#     #     # Where legal_mask==0, we want -inf; where legal_mask==1, we want 0 (no change)
#     #     masked_logits = raw_logits.masked_fill(legal_mask < 0.5, -1e8)
        
#     #     value = self.value_head(hidden).squeeze(-1)
        
#     #     return masked_logits, value

#     def forward(self, obs, state=None):
#         # This forward is used during non-LSTM inference
#         # Extract components
#         board_state = obs[:, :1344]
#         legal_mask = obs[:, 1472:3440]  # 1968 UCI actions
        
#         # Encode board with MLP
#         board_features = self.board_encoder(board_state)
#         hidden = self.combiner(board_features)
        
#         # Decode with proper masking
#         logits, value = self.decode_actions(hidden, legal_mask)

#         return logits, value


class ChessLSTM(Recurrent):
    def __init__(self, env, policy, input_size, hidden_size):
        super().__init__(env, policy, input_size, hidden_size)

class ChessRecurrent(nn.Module):
    """Optimized MLP encoder/decoder for chess observations with UCI action space."""
    def __init__(self, env=None, hidden_size=256, num_actions=1968, **kwargs):
        super().__init__()
        # Simple MLP encoder for flattened chess board (much faster than CNN)
        self.board_encoder = nn.Sequential(
            nn.Linear(1472, 512),  # 23 * 8 * 8 = 1472
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        # Combiner projects board_features to hidden state
        self.combiner = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.ReLU()
        )
        # Output heads for UCI action space
        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.is_continuous = False
        self.is_multidiscrete = False
        self.action_size = num_actions

    def encode_observations(self, obs, state=None):
        # For LSTM compatibility, we only encode board features here
        # Handle both 1D and 2D tensor inputs
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        board_state = obs[:, :1472].float()
        board_features = self.board_encoder(board_state)
        hidden = self.combiner(board_features)
        return hidden

    def decode_actions(self, hidden, legal_mask):
        # Get raw logits for UCI actions
        raw_logits = self.policy_head(hidden)
        
        # Apply masking: set invalid actions to -inf
        masked_logits = raw_logits.masked_fill(legal_mask < 0.5, float('-inf'))
        
        value = self.value_head(hidden).squeeze(-1)
        
        return masked_logits, value


    def forward(self, obs, state=None):
        # Non-LSTM inference path
        hidden = self.encode_observations(obs[:, :1472])
        # Convert sparse mask to dense format using GPU operations
        # Fixed import
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'chess'))
        import sparse_utils
        sparse_to_dense_gpu = sparse_utils.sparse_to_dense_gpu
        legal_mask = sparse_to_dense_gpu(obs)
        logits, value = self.decode_actions(hidden, legal_mask)
        return logits, value

def Policy(env, **kwargs):
    """Factory returning the appropriate policy for the given environment."""
    hidden_size = kwargs.pop('hidden_size', 256)
    base = ChessRecurrent(env, hidden_size=hidden_size)
    return ChessLSTM(env, base, input_size=hidden_size, hidden_size=hidden_size)

        
class Tetris(nn.Module):
    def __init__(
        self, 
        env, 
        cnn_channels=32,
        input_size=128,
        hidden_size=128,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cnn_channels =  cnn_channels   
        self.n_cols = env.n_cols
        self.n_rows = env.n_rows
        self.scalar_input_size = (6 + 7 * (env.deck_size + 1))
        self.flat_conv_size = cnn_channels * 3 * 10
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

        self.conv_grid = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(2, cnn_channels, kernel_size=(5, 3), stride=(2,1), padding=(2,1))),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(cnn_channels, cnn_channels, kernel_size=(5, 3), stride=(2,1), padding=(2,1))),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(cnn_channels, cnn_channels, kernel_size=(5, 5), stride=(2,1), padding=(2,2))),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(self.flat_conv_size, input_size)),
        )

        self.fc_scalar = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.scalar_input_size, input_size)),
            nn.ReLU(),
        )

        self.proj = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(2 * input_size, hidden_size)),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 7), std=0.01),
            nn.Flatten()
        )

        self.value_fn = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1)),
            nn.ReLU(),
        )

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations) 
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        B = observations.shape[0]
        grid_info = observations[:, 0:(self.n_cols * self.n_rows)].view(B, self.n_rows, self.n_cols)  # (B, n_rows, n_cols)
        grid_info = torch.stack([(grid_info == 1).float(), (grid_info == 2).float()], dim=1)  # (B, 2, n_rows, n_cols)
        scalar_info = observations[:, (self.n_cols * self.n_rows):(self.n_cols * self.n_rows + self.scalar_input_size)].float()

        grid_feat = self.conv_grid(grid_info)  # (B, input_size)
        scalar_feat = self.fc_scalar(scalar_info)  # (B, input_size)

        combined = torch.cat([grid_feat, scalar_feat], dim=-1)  # (B, 2 * input_size)
        features = self.proj(combined)  # (B, hidden_size)
        return features

    def decode_actions(self, hidden):
        action = self.actor(hidden)  # (B, 4 * n_cols)
        value = self.value_fn(hidden)  # (B, 1)
        return action, value

class Drone(nn.Module):
    ''' Drone policy. Flattens obs and applies a linear layer.
    '''
    def __init__(self, env, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_multidiscrete = isinstance(env.single_action_space,
                pufferlib.spaces.MultiDiscrete)
        self.is_continuous = isinstance(env.single_action_space,
                pufferlib.spaces.Box)
        try:
            self.is_dict_obs = isinstance(env.env.observation_space, pufferlib.spaces.Dict) 
        except:
            self.is_dict_obs = isinstance(env.observation_space, pufferlib.spaces.Dict) 

        if self.is_dict_obs:
            self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
            input_size = int(sum(np.prod(v.shape) for v in env.env.observation_space.values()))
            self.encoder = nn.Linear(input_size, self.hidden_size)
        else:
            self.encoder = torch.nn.Sequential(
                nn.Linear(np.prod(env.single_observation_space.shape), hidden_size),
                nn.GELU(),
            )

        if self.is_multidiscrete:
            self.action_nvec = tuple(env.single_action_space.nvec)
            self.decoder = pufferlib.pytorch.layer_init(
                    nn.Linear(hidden_size, sum(self.action_nvec)), std=0.01)
        elif not self.is_continuous:
            self.decoder = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        else:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.decoder_logstd = nn.Parameter(torch.zeros(
                1, env.single_action_space.shape[0]))

        self.value = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def _validate_chess_torch_inputs(self, observations, location):
        """COLOR MONITORING: Validates chess inputs at torch.py policy level"""
        if observations is None or observations.numel() == 0:
            print(f"[MONITOR_FATAL] Torch.py: Empty observations at {location}")
            print(f"  Chess policy received empty observation tensor")
            print(f"  FIX: Check models.py policy forwarding")
            exit(1)
            
        if observations.shape[-1] == 1537:  # Chess observations (sparse format)
            batch_size = observations.shape[0]
            board_part = observations[:, :1472]
            # Sparse mask: num_legal_moves at 1472, action_ids from 1473 onwards
            num_legal_moves = observations[:, 1472]
            
            # Validate content structure
            board_sums = board_part.sum(dim=1)
            
            if (board_sums < 1.0).any() or (num_legal_moves < 0).any() or (num_legal_moves > 64).any():
                print(f"[MONITOR_FATAL] Torch.py: Invalid chess observations at {location}")
                print(f"  Board sums: {board_sums}")
                print(f"  Num legal moves: {num_legal_moves}")
                print(f"  FIX: Check observation encoding in chess.h")
                exit(1)
                
            print(f"[MONITOR_OK] Torch.py: Chess policy inputs valid at {location} "
                  f"(batch={batch_size}, board_range=[{board_sums.min():.1f},{board_sums.max():.1f}], "
                  f"legal_moves_range=[{num_legal_moves.min():.0f},{num_legal_moves.max():.0f}])")

    def _validate_chess_torch_outputs(self, logits, values, location):
        """COLOR MONITORING: Validates chess outputs at torch.py policy level"""
        if logits is None or logits.numel() == 0:
            print(f"[MONITOR_FATAL] Torch.py: Empty logits at {location}")
            print(f"  Chess policy generated empty logits")
            print(f"  FIX: Check policy_head in ChessRecurrent model")
            exit(1)
            
        if logits.shape[-1] == 1968:  # Chess UCI actions
            batch_size = logits.shape[0]
            
            # Validate logits quality - allow -inf for action masking but not +inf or NaN
            if torch.isnan(logits).any() or (logits == float('inf')).any():
                nan_count = torch.isnan(logits).sum().item()
                pos_inf_count = (logits == float('inf')).sum().item()
                print(f"[MONITOR_FATAL] Torch.py: Invalid logits at {location}")
                print(f"  NaN values: {nan_count}, Positive Inf values: {pos_inf_count}")
                print(f"  FIX: Check neural network training or input preprocessing")
                exit(1)
                
            # Check for no legal actions (all logits == -inf except perhaps some)
            if (logits > -1e10).sum() == 0:  # No finite logits
                print(f"[MONITOR_WARNING] Torch.py: No legal actions at {location}")
                print(f"  All logits masked - possible game over or bug")
                
            # Check for reasonable logit distribution
            logit_min, logit_max = logits.min().item(), logits.max().item()
            masked_count = (logits <= -1e7).sum().item()  # Count masked actions
            
            print(f"[MONITOR_OK] Torch.py: Chess policy outputs valid at {location} "
                  f"(batch={batch_size}, logit_range=[{logit_min:.1f},{logit_max:.1f}], "
                  f"masked_actions={masked_count}, value_range=[{values.min():.3f},{values.max():.3f}])")

    def forward_eval(self, observations, state=None):
        # --- COLOR MONITORING: Validate chess-specific policy inputs ---
        self._validate_chess_torch_inputs(observations, "ChessRecurrent.forward_eval() input")
        
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        
        # --- COLOR MONITORING: Validate chess-specific policy outputs ---
        self._validate_chess_torch_outputs(logits, values, "ChessRecurrent.forward_eval() output")
        
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        '''Encodes a batch of observations into hidden states. Assumes
        no time dimension (handled by LSTM wrappers).'''
        batch_size = observations.shape[0]
        if self.is_dict_obs:
            observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
            observations = torch.cat([v.view(batch_size, -1) for v in observations.values()], dim=1)
        else: 
            observations = observations.view(batch_size, -1)
        return self.encoder(observations.float())

    def decode_actions(self, hidden):
        '''Decodes a batch of hidden states into (multi)discrete actions.
        Assumes no time dimension (handled by LSTM wrappers).'''
        if self.is_multidiscrete:
            logits = self.decoder(hidden).split(self.action_nvec, dim=1)
        elif self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            logits = torch.distributions.Normal(mean, std)
        else:
            logits = self.decoder(hidden)

        values = self.value(hidden)
        return logits, values

def Policy(env, **kwargs):
    """Factory returning the appropriate policy for the given environment.

    If the observation matches the chess shape (3312,) we build a
    ChessRecurrent model and wrap it with the generic LSTMWrapper so that
    temporal batching works out of the box. Otherwise we fall back to the
    legacy Default policy so existing non-chess environments remain
    functional.
    """
    # Chess: obs = 1472 board planes + 1968 legal-move mask
    try:
        is_chess = (len(env.single_observation_space.shape) == 1 and
                    env.single_observation_space.shape[0] == 1537)
    except AttributeError:
        is_chess = False

    if is_chess:
        hidden_size = kwargs.pop('hidden_size', 256)
        base = ChessRecurrent(env, hidden_size=hidden_size)
        return Recurrent(env, base, input_size=hidden_size, hidden_size=hidden_size)
    else:
        return _OldDefault(env, **kwargs)

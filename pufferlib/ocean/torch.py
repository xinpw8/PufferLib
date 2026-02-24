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

from pufferlib.models import Default as Policy
from pufferlib.models import MinGRU, Mamba, GRU, MinGRULayer
from pufferlib.models import DefaultEncoder, DefaultDecoder
from pufferlib.models import Convolutional as Conv
Recurrent = pufferlib.models.LSTMWrapper
from pufferlib.pytorch import layer_init, _nativize_dtype, nativize_tensor
import numpy as np


class ChessSeven(nn.Module):
    def __init__(self, env, square_dim=64, proj_dim=8, hidden_size=256,
                 embed_dim=32, use_action_masking=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        self.use_action_masking = bool(use_action_masking)
        self.num_actions = env.single_action_space.n

        sqs = torch.arange(64, dtype=torch.float32)
        r, f = sqs // 8, sqs % 8
        diag = (r + f) / 14.0
        anti = (r - f + 7) / 14.0
        cdist = (torch.where(r < 4, 3 - r, r - 4) + torch.where(f < 4, 3 - f, f - 4)) / 6.0
        sq_color = ((r + f) % 2).float()
        square_geo_planes = torch.stack([diag, anti, cdist, sq_color], dim=0).view(1, 4, 8, 8)
        self.register_buffer('square_geo_planes', square_geo_planes)

        # 15 spatial channels from obs + 4 geometric = 19
        self.square_embed = layer_init(nn.Conv2d(19, square_dim, kernel_size=1))
        self.channel_proj = layer_init(nn.Conv2d(square_dim, proj_dim, kernel_size=1))
        self.spatial_mix = layer_init(nn.Conv2d(
            proj_dim, proj_dim, kernel_size=3, padding=1, groups=proj_dim))

        if embed_dim % 2 != 0:
            raise ValueError(f'embed_dim must be even, got {embed_dim}')
        self.side_embed = nn.Embedding(2, embed_dim // 2)
        self.castle_embed = nn.Embedding(16, embed_dim)
        self.ep_embed = nn.Embedding(65, embed_dim)
        self.phase_embed = nn.Embedding(2, embed_dim // 2)

        board_flat = 64 * proj_dim + 32
        total_features = board_flat + (3 * embed_dim) + 5

        self.proj = nn.Sequential(
            layer_init(nn.Linear(total_features, hidden_size)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(hidden_size, self.num_actions), std=0.01)
        self.value_head = layer_init(nn.Linear(hidden_size, 1), std=1)
        self.current_mask = None

    def encode_observations(self, observations, state=None):
        B = observations.shape[0]
        obs = observations

        # Spatial features from 1082-byte obs layout
        board = obs[:, :768].float().view(B, 12, 8, 8)
        selected = obs[:, 853:917].float().view(B, 1, 8, 8)
        valid_pieces_sp = obs[:, 917:981].float().view(B, 1, 8, 8)
        valid_dests_sp = obs[:, 981:1045].float().view(B, 1, 8, 8)
        geo = self.square_geo_planes.expand(B, -1, -1, -1)
        x = torch.cat([board, selected, valid_pieces_sp, valid_dests_sp, geo], dim=1)

        x = F.relu(self.square_embed(x))
        x = F.relu(self.channel_proj(x))
        x = x + F.relu(self.spatial_mix(x))
        board_features = x.flatten(1)

        promos_mask = obs[:, 1045:1077] > 0
        promos = promos_mask.float()

        side_features = self.side_embed(obs[:, 768:770].argmax(1))
        castle_features = self.castle_embed(obs[:, 770:786].argmax(1))
        ep_features = self.ep_embed(obs[:, 786:851].argmax(1))
        phase_features = self.phase_embed(obs[:, 851:853].argmax(1))
        scalars = obs[:, 1077:1082].float() / 255.0

        if self.use_action_masking:
            pick_phase = obs[:, 852] > 0
            pass_valid = obs[:, 1081] > 0
            valid_pieces_mask = obs[:, 917:981] > 0
            valid_dests_mask = obs[:, 981:1045] > 0

            mask_squares = torch.where(pick_phase.unsqueeze(1), valid_dests_mask, valid_pieces_mask)
            full_mask = torch.cat([mask_squares, promos_mask, pass_valid.unsqueeze(1)], dim=1)
            full_mask[:, :-1] = full_mask[:, :-1] & (~pass_valid.unsqueeze(1))
            all_masked = ~full_mask.any(dim=1, keepdim=True)
            full_mask = full_mask | all_masked
            self.current_mask = full_mask
        else:
            self.current_mask = None

        x = torch.cat([board_features, promos,
                        side_features, castle_features, ep_features, phase_features,
                        scalars], dim=1)
        x = self.proj(x)
        return x

    def decode_actions(self, hidden, state=None):
        logits = self.actor(hidden)
        if self.use_action_masking and self.current_mask is not None:
            logits.masked_fill_(~self.current_mask, -1e8)
        value = self.value_head(hidden)
        return logits, value

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations, state)
        logits, value = self.decode_actions(hidden, state)
        return logits, value

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)


class ChessTwo(nn.Module):
    def __init__(self, env, cnn_channels=256, hidden_size=512, embed_dim=32, use_action_masking=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        self.use_action_masking = bool(use_action_masking)
        self.num_actions = env.single_action_space.n

        self.conv1 = layer_init(nn.Conv2d(16, cnn_channels, kernel_size=3, stride=1, padding=1))
        self.conv2 = layer_init(nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, stride=1, padding=1))
        self.conv3 = layer_init(nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, stride=1, padding=1))
        self.conv4 = layer_init(nn.Conv2d(cnn_channels, hidden_size, kernel_size=3, stride=1, padding=1))

        with torch.no_grad():
            dummy = torch.zeros(1, 16, 8, 8)
            x = nn.ReLU()(self.conv1(dummy))
            residual = x
            x = nn.ReLU()(self.conv2(x))
            x = self.conv3(x)
            x = x + residual
            x = nn.ReLU()(x)
            x = self.conv4(x)
            x = nn.ReLU()(x)
            cnn_flat_size = x.flatten(1).shape[1]

        self.side_embed = nn.Embedding(2, embed_dim)
        self.castle_embed = nn.Embedding(16, embed_dim)
        self.ep_embed = nn.Embedding(65, embed_dim)
        self.phase_embed = nn.Embedding(2, embed_dim)

        self.scalar_size = 5
        self.scalar_layer = nn.Sequential(
            layer_init(nn.Linear(self.scalar_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )
        total_features = cnn_flat_size + 4 * embed_dim + hidden_size

        self.proj = nn.Sequential(
            layer_init(nn.Linear(total_features, hidden_size)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(hidden_size, self.num_actions), std=0.01)
        self.value_head = layer_init(nn.Linear(hidden_size, 1), std=1)
        self.current_mask = None

    def encode_observations(self, observations, state=None):
        B = observations.shape[0]
        obs = observations.float()

        board = obs[:, :768].view(B, 12, 8, 8)
        selected_piece = obs[:, 853:917].view(B, 1, 8, 8)
        valid_pieces = obs[:, 917:981].view(B, 1, 8, 8)
        valid_dests = obs[:, 981:1045].view(B, 1, 8, 8)
        valid_promos = obs[:, 1045:1077].view(B, 1, 4, 8)
        valid_promos_padded = F.pad(valid_promos, (0, 0, 0, 4), value=0).view(B, 1, 8, 8)

        spatial_input = torch.cat([
            board, selected_piece, valid_pieces, valid_dests, valid_promos_padded
        ], dim=1)

        x = self.conv1(spatial_input)
        x = nn.ReLU()(x)
        residual = x
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = x + residual
        x = nn.ReLU()(x)
        x = self.conv4(x)
        x = nn.ReLU()(x)
        spatial_features = x.flatten(1)

        side_features = self.side_embed(obs[:, 768:770].argmax(dim=1))
        castle_features = self.castle_embed(obs[:, 770:786].argmax(dim=1))
        ep_features = self.ep_embed(obs[:, 786:851].argmax(dim=1))
        phase_features = self.phase_embed(obs[:, 851:853].argmax(dim=1))

        self_check = obs[:, 1077:1078] / 255.0
        opp_check = obs[:, 1078:1079] / 255.0
        rule50_scalar = obs[:, 1079:1080] / 255.0
        repetition_scalar = obs[:, 1080:1081] / 255.0
        pass_valid = obs[:, 1081:1082] / 255.0
        scalars = torch.cat([self_check, opp_check, rule50_scalar, repetition_scalar, pass_valid], dim=1)
        scalars = self.scalar_layer(scalars)

        if self.use_action_masking:
            pick_phase = observations[:, 852] > 0
            pass_valid_mask = observations[:, 1081] > 0
            valid_pieces_mask = observations[:, 917:981] > 0
            valid_dests_mask = observations[:, 981:1045] > 0
            promos_mask = observations[:, 1045:1077] > 0

            mask_squares = torch.where(pick_phase.unsqueeze(1), valid_dests_mask, valid_pieces_mask)
            full_mask = torch.cat([mask_squares, promos_mask, pass_valid_mask.unsqueeze(1)], dim=1)
            full_mask[:, :-1] = full_mask[:, :-1] & (~pass_valid_mask.unsqueeze(1))
            all_masked = ~full_mask.any(dim=1, keepdim=True)
            full_mask = full_mask | all_masked
            self.current_mask = full_mask
        else:
            self.current_mask = None

        x = torch.cat([spatial_features, side_features, castle_features, ep_features, phase_features, scalars], dim=1)
        x = self.proj(x)
        return x

    def decode_actions(self, hidden, state=None):
        logits = self.actor(hidden)
        if self.use_action_masking and self.current_mask is not None:
            logits.masked_fill_(~self.current_mask, -1e8)
        value = self.value_head(hidden)
        return logits, value

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations, state)
        logits, value = self.decode_actions(hidden, state)
        return logits, value

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)


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

class SnakeEncoder(nn.Module):
    def __init__(self, env, hidden_size=128):
        super().__init__()
        num_obs = np.prod(env.single_observation_space.shape)
        dtype = env.single_observation_space.dtype

        self.dtype = dtype
        self.encoder = pufferlib.pytorch.layer_init(nn.Linear(8*num_obs, hidden_size))

    def forward(self, observations):
        batch_size = observations.shape[0]
        observations = F.one_hot(observations.long(), 8).view(-1, 11*11*8).float()
        hidden = self.encoder(observations.float())
        return F.gelu(hidden)


class Snake(nn.Module):
    def __init__(self, env, hidden_size=128, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = hidden_size
        self.num_layers = num_layers
        self.obs_shape = env.single_observation_space.shape
        self.encoder = SnakeEncoder(env, hidden_size)
        self.decoder = DefaultDecoder(env, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.cell = nn.ModuleList([torch.nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers)])

        for i in range(num_layers):
            cell = self.cell[i]

            w_ih = getattr(self.lstm, f'weight_ih_l{i}')
            w_hh = getattr(self.lstm, f'weight_hh_l{i}')
            b_ih = getattr(self.lstm, f'bias_ih_l{i}')
            b_hh = getattr(self.lstm, f'bias_hh_l{i}')

            nn.init.orthogonal_(w_ih, 1.0)
            nn.init.orthogonal_(w_hh, 1.0)
            b_ih.data.zero_()
            b_hh.data.zero_()

            cell.weight_ih = w_ih
            cell.weight_hh = w_hh
            cell.bias_ih = b_ih
            cell.bias_hh = b_hh

    def initial_state(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c

    def forward_eval(self, x, state):
        '''Forward function for inference. 3x faster than using LSTM directly'''
        assert state[0].shape[1] == state[1].shape[1] == x.shape[0], 'LSTM state must be (h, c)'
        h = self.encoder(x)
        lstm_h, lstm_c = state
        for i in range(self.num_layers):
            h, c = self.cell[i](h, (lstm_h[i], lstm_c[i]))
            lstm_h[i] = h
            lstm_c[i] = c

        logits, values = self.decoder(h)
        return logits, values, (lstm_h, lstm_c)

    def forward(self, x):
        '''Forward function for training. Uses LSTM for fast time-batching'''
        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        assert x_shape[-space_n:] == space_shape, f'Invalid input tensor shape {x.shape} != {space_shape}'

        B, TT = x_shape[:2]
        x = x.reshape(B*TT, *space_shape)
        h = self.encoder(x)
        assert h.shape == (B*TT, self.input_size)
        h = h.reshape(B, TT, self.input_size)

        h = h.transpose(0, 1)
        h, (lstm_h, lstm_c) = self.lstm.forward(h)
        h = h.transpose(0, 1)

        flat_hidden = h.reshape(B*TT, self.hidden_size)
        logits, values = self.decoder(flat_hidden)
        values = values.reshape(B, TT)
        return logits, values



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
    def __init__(self, env, policy, hidden_size: int = 512, **kwargs):
        super().__init__(env, policy, hidden_size)


class ImpulseWarsPolicy(nn.Module):
    def __init__(
        self,
        env: pufferlib.PufferEnv,
        cnn_channels: int = 64,
        weapon_type_embedding_dims: int = 2,
        hidden_size: int = 512,
        batch_size: int = 131_072,
        num_drones: int = 2,
        continuous: bool = False,
        is_training: bool = True,
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
        discreteOffsets = torch.tensor([0] + list(np.cumsum(self.discreteFactors)[:-1])).view(
            1, -1
        )
        self.register_buffer("discreteOffsets", discreteOffsets, persistent=False)
        self.discreteMultihotDim = self.discreteFactors.sum()

        multihotBuffer = torch.zeros(batch_size, self.discreteMultihotDim)
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
            layer_init(nn.Linear(featuresSize, hidden_size)),
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

class Drive(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(7, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Linear(input_size, input_size))
        )
        max_road_objects = 13
        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(max_road_objects, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Linear(input_size, input_size))
        )
        max_partner_objects = 7
        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(max_partner_objects, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Linear(input_size, input_size))
        )


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
        ego_dim = 7
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

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
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


class G2048(nn.Module):
    def __init__(self, env, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        self.embed_dim = int(np.ceil(33**0.25))
        self.num_grid_cell = 4*4
        self.num_obs = self.num_grid_cell * self.embed_dim

        self.value_embed = torch.nn.Embedding(18, self.embed_dim)
        self.pos_embed = torch.nn.Embedding(self.num_grid_cell, self.embed_dim)

        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(self.num_obs, 2 * hidden_size)),
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(2 * hidden_size, hidden_size)),
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.GELU(),
        )

        num_atns = env.single_action_space.n
        self.decoder = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, num_atns), std=0.01),
        )
        self.value = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        value_obs = self.value_embed(observations.long())
        pos_obs = self.pos_embed.weight.expand(*value_obs.shape)
        grid_obs = (value_obs + pos_obs).flatten(1)
        return self.encoder(grid_obs)

    def decode_actions(self, hidden):
        logits = self.decoder(hidden)
        values = self.value(hidden)
        return logits, values

class G2048LSTM(nn.Module):
    def __init__(self, env, hidden_size=128, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = hidden_size
        self.num_layers = num_layers
        self.obs_shape = env.single_observation_space.shape

        self.g2048 = G2048(env, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.cell = nn.ModuleList([torch.nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers)])

        for i in range(num_layers):
            cell = self.cell[i]

            w_ih = getattr(self.lstm, f'weight_ih_l{i}')
            w_hh = getattr(self.lstm, f'weight_hh_l{i}')
            b_ih = getattr(self.lstm, f'bias_ih_l{i}')
            b_hh = getattr(self.lstm, f'bias_hh_l{i}')

            nn.init.orthogonal_(w_ih, 1.0)
            nn.init.orthogonal_(w_hh, 1.0)
            b_ih.data.zero_()
            b_hh.data.zero_()

            cell.weight_ih = w_ih
            cell.weight_hh = w_hh
            cell.bias_ih = b_ih
            cell.bias_hh = b_hh

    def initial_state(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c

    def forward_eval(self, x, state):
        '''Forward function for inference. 3x faster than using LSTM directly'''
        assert state[0].shape[1] == state[1].shape[1] == x.shape[0], 'LSTM state must be (h, c)'
        h = self.g2048.encode_observations(x)
        lstm_h, lstm_c = state
        for i in range(self.num_layers):
            h, c = self.cell[i](h, (lstm_h[i], lstm_c[i]))
            lstm_h[i] = h
            lstm_c[i] = c

        logits, values = self.g2048.decode_actions(h)
        return logits, values, (lstm_h, lstm_c)

    def forward(self, x):
        '''Forward function for training. Uses LSTM for fast time-batching'''
        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        assert x_shape[-space_n:] == space_shape, f'Invalid input tensor shape {x.shape} != {space_shape}'

        B, TT = x_shape[:2]
        x = x.reshape(B*TT, *space_shape)
        h = self.g2048.encode_observations(x)
        assert h.shape == (B*TT, self.input_size)
        h = h.reshape(B, TT, self.input_size)

        h = h.transpose(0, 1)
        h, (lstm_h, lstm_c) = self.lstm.forward(h)
        h = h.transpose(0, 1)

        flat_hidden = h.reshape(B*TT, self.hidden_size)
        logits, values = self.g2048.decode_actions(flat_hidden)
        values = values.reshape(B, TT)
        return logits, values

class G2048MinGRU(nn.Module):
    def __init__(self, env, hidden_size=128, num_layers=1, expansion_factor=2, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = hidden_size
        self.expansion_factor = expansion_factor
        self.obs_shape = env.single_observation_space.shape

        self.g2048 = G2048(env, hidden_size)
        self.expansion_factor = expansion_factor
        self.num_layers = num_layers
        self.mingru = nn.ModuleList([MinGRULayer(hidden_size, expansion_factor) for _ in range(num_layers)])

    def initial_state(self, batch_size, device):
        state = torch.zeros(self.num_layers, batch_size, self.hidden_size*self.expansion_factor, device=device)
        return (state,)

    def forward_eval(self, x, state):
        state = state[0]
        assert state.shape[1] == x.shape[0]
        h = self.g2048.encode_observations(x)
        h = h.unsqueeze(1)
        state = state.unsqueeze(2)
        state_out = []
        for i in range(self.num_layers):
            h, s = self.mingru[i](h, state[i])
            state_out.append(s)

        h = h.squeeze(1)
        state = torch.stack(state_out, 0).squeeze(2)
        logits, values = self.g2048.decode_actions(h)
        return logits, values, (state,)

    def forward(self, x):
        '''Forward function for training. Uses LSTM for fast time-batching'''
        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        assert x_shape[-space_n:] == space_shape, f'Invalid input tensor shape {x.shape} != {space_shape}'

        B, TT = x_shape[:2]
        x = x.reshape(B*TT, *space_shape)
        h = self.g2048.encode_observations(x)
        assert h.shape == (B*TT, self.input_size)
        h = h.reshape(B, TT, self.input_size)

        state = self.initial_state(B, h.device)[0].unsqueeze(2)
        for i in range(self.num_layers):
            h, _ = self.mingru[i](h, state[i])

        flat_hidden = h.reshape(B*TT, self.hidden_size)
        logits, values = self.g2048.decode_actions(flat_hidden)
        values = values.reshape(B, TT)
        return logits, values

class NMMO3MinGRU(nn.Module):
    def __init__(self, env, hidden_size=128, num_layers=1, expansion_factor=2, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = hidden_size
        self.expansion_factor = expansion_factor
        self.obs_shape = env.single_observation_space.shape

        self.nmmo3 = NMMO3(env, hidden_size)
        self.expansion_factor = expansion_factor
        self.num_layers = num_layers
        self.mingru = nn.ModuleList([MinGRULayer(hidden_size, expansion_factor) for _ in range(num_layers)])

    def initial_state(self, batch_size, device):
        state = torch.zeros(self.num_layers, batch_size, self.hidden_size*self.expansion_factor, device=device)
        return (state,)

    def forward_eval(self, x, state):
        state = state[0]
        assert state.shape[1] == x.shape[0]
        h = self.nmmo3.encode_observations(x)
        h = h.unsqueeze(1)
        state = state.unsqueeze(2)
        state_out = []
        for i in range(self.num_layers):
            h, s = self.mingru[i](h, state[i])
            state_out.append(s)

        h = h.squeeze(1)
        state = torch.stack(state_out, 0).squeeze(2)
        logits, values = self.nmmo3.decode_actions(h)
        return logits, values, (state,)

    def forward(self, x):
        '''Forward function for training. Uses LSTM for fast time-batching'''
        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        assert x_shape[-space_n:] == space_shape, f'Invalid input tensor shape {x.shape} != {space_shape}'

        B, TT = x_shape[:2]
        x = x.reshape(B*TT, *space_shape)
        h = self.nmmo3.encode_observations(x)
        assert h.shape == (B*TT, self.input_size)
        h = h.reshape(B, TT, self.input_size)

        state = self.initial_state(B, h.device)[0].unsqueeze(2)
        for i in range(self.num_layers):
            h, _ = self.mingru[i](h, state[i])

        flat_hidden = h.reshape(B*TT, self.hidden_size)
        logits, values = self.nmmo3.decode_actions(flat_hidden)
        values = values.reshape(B, TT)
        return logits, values

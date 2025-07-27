import os
from pdb import set_trace as T
import numpy as np

import torch
import torch.nn as nn

import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.spaces


class Default(nn.Module):
    '''Default PyTorch policy. Flattens obs and applies a linear layer.

    PufferLib is not a framework. It does not enforce a base class.
    You can use any PyTorch policy that returns actions and values.
    We structure our forward methods as encode_observations and decode_actions
    to make it easier to wrap policies with LSTMs. You can do that and use
    our LSTM wrapper or implement your own. To port an existing policy
    for use with our LSTM wrapper, simply put everything from forward() before
    the recurrent cell into encode_observations and put everything after
    into decode_actions.
    '''
    def __init__(self, env, hidden_size=256):
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

        # Check if this is a chess environment by examining observation space
        try:
            obs_shape = env.single_observation_space.shape
            self.is_chess = (len(obs_shape) == 1 and obs_shape[0] == 1537)  # 1472 board + 1 + 64 sparse mask
        except:
            self.is_chess = False

        if self.is_chess:
            # Chess-specific architecture
            # Board features: 1472 dims (23 channels × 8×8)
            # Legal mask: 1968 dims
            self.board_encoder = nn.Sequential(
                nn.Linear(1472, 512),
                nn.ReLU(),
                nn.Linear(512, hidden_size),
                nn.ReLU()
            )
            input_size = hidden_size
        elif self.is_dict_obs:
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
            if self.is_chess:
                # Chess has 1968 possible actions
                self.decoder = pufferlib.pytorch.layer_init(
                    nn.Linear(hidden_size, 1968), std=0.01)
            else:
                self.decoder = pufferlib.pytorch.layer_init(
                    nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        else:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.decoder_logstd = nn.Parameter(torch.zeros(
                1, env.single_action_space.shape[0]))

        self.value = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def _validate_chess_policy_inputs(self, observations, location):
        """COLOR MONITORING: Validates chess inputs at policy level"""
        if observations is None or observations.numel() == 0:
            print(f"[MONITOR_FATAL] Models.py: Empty observations at {location}")
            print(f"  Policy received empty observation tensor")
            print(f"  FIX: Check pufferl.py observation processing")
            exit(1)
            
        if observations.shape[-1] == 1537:  # Chess observations (sparse format)
            batch_size = observations.shape[0]
            board_part = observations[:, :1472]
            # Sparse mask: num_legal_moves at 1472, action_ids from 1473 onwards
            num_legal_moves = observations[:, 1472]
            
            board_sums = board_part.sum(dim=1)
            
            if (board_sums < 1.0).any() or (num_legal_moves < 0).any() or (num_legal_moves > 64).any():
                print(f"[MONITOR_FATAL] Models.py: Invalid chess observations at {location}")
                print(f"  Board sums: {board_sums}")
                print(f"  Num legal moves: {num_legal_moves}")
                print(f"  FIX: Check observation generation pipeline")
                exit(1)
                
            print(f"[MONITOR_OK] Models.py: Chess policy inputs valid at {location} "
                  f"(batch={batch_size}, board_range=[{board_sums.min():.1f},{board_sums.max():.1f}], "
                  f"legal_moves_range=[{num_legal_moves.min():.0f},{num_legal_moves.max():.0f}])")

    def _validate_chess_policy_outputs(self, logits, value, location):
        """COLOR MONITORING: Validates chess outputs at policy level"""
        if logits is None or (hasattr(logits, 'numel') and logits.numel() == 0):
            print(f"[MONITOR_FATAL] Models.py: Empty logits at {location}")
            print(f"  Policy generated empty logits tensor")
            print(f"  FIX: Check decode_actions() in chess policy")
            exit(1)
            
        if hasattr(logits, 'shape') and logits.shape[-1] == 1968:  # Chess UCI actions
            batch_size = logits.shape[0]
            
            # Check for invalid logits (NaN, inf)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                nan_count = torch.isnan(logits).sum().item()
                inf_count = torch.isinf(logits).sum().item()
                print(f"[MONITOR_FATAL] Models.py: Invalid logits at {location}")
                print(f"  NaN values: {nan_count}, Inf values: {inf_count}")
                print(f"  FIX: Check neural network weights or action masking")
                exit(1)
                
            # Check logit range (should be reasonable for softmax)
            logit_min, logit_max = logits.min().item(), logits.max().item()
            if logit_max - logit_min > 100:  # Extreme range indicates masking issues
                print(f"[MONITOR_FATAL] Models.py: Extreme logit range at {location}")
                print(f"  Logit range: [{logit_min:.1f}, {logit_max:.1f}]")
                print(f"  This suggests action masking problems")
                print(f"  FIX: Check legal move mask application in chess policy")
                exit(1)
                
            print(f"[MONITOR_OK] Models.py: Chess policy outputs valid at {location} "
                  f"(batch={batch_size}, logit_range=[{logit_min:.1f},{logit_max:.1f}], "
                  f"value_range=[{value.min():.3f},{value.max():.3f}])")


    def forward_eval(self, observations, state=None):
        if not hasattr(self, "_dbg_printed"):
            print("DEBUG: Default policy forward_eval called – this should NOT happen for chess.")
            self._dbg_printed = True
            
        # --- COLOR MONITORING: Validate policy inputs ---
        self._validate_chess_policy_inputs(observations, "forward_eval() input")
            
        hidden = self.encode_observations(observations, state=state)
        if self.is_chess:
            # Convert sparse mask to dense format using GPU-optimized operations
            # Fixed import
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), 'ocean', 'chess'))
            import sparse_utils
            sparse_to_dense_gpu = sparse_utils.sparse_to_dense_gpu
            legal_mask = sparse_to_dense_gpu(observations)
            logits, values = self.decode_actions(hidden, legal_mask)
        else:
            logits, values = self.decode_actions(hidden)
        
        # --- COLOR MONITORING: Validate policy outputs ---
        self._validate_chess_policy_outputs(logits, values, "forward_eval() output")
        
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        '''Encodes a batch of observations into hidden states. Assumes
        no time dimension (handled by LSTM wrappers).'''
        batch_size = observations.shape[0]
        
        if self.is_chess:
            # Chess: encode only board features (first 1472 dims)
            board_features = observations[:, :1472]
            return self.board_encoder(board_features.float())
        elif self.is_dict_obs:
            observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
            observations = torch.cat([v.view(batch_size, -1) for v in observations.values()], dim=1)
            return self.encoder(observations.float())
        else: 
            observations = observations.view(batch_size, -1)
            return self.encoder(observations.float())

    def decode_actions(self, hidden, legal_mask=None):
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
            raw_logits = self.decoder(hidden)
            
            if self.is_chess and legal_mask is not None:
                # Apply action masking as described in the PPO paper
                # Set invalid actions to very negative logits (-1e8)
                logits = raw_logits.masked_fill(legal_mask < 0.5, -1e8)
                
                # Handle empty legal masks (dual-agent mode when it's not this agent's turn)
                # In dual-agent chess, agents receive empty legal masks when it's not their turn.
                # This is expected behavior - the agent shouldn't act, and the training framework
                # will ignore this agent's output anyway.
                if torch.sum(legal_mask) == 0:
                    # All actions are already masked to -1e8, which is correct
                    pass
            else:
                logits = raw_logits

        values = self.value(hidden)
        return logits, values


class LSTMWrapper(nn.Module):
    def __init__(self, env, policy, input_size=128, hidden_size=128):
        '''Wraps your policy with an LSTM.'''
        super().__init__()
        self.obs_shape = env.single_observation_space.shape
        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTM layer for fast parallel training
        self.lstm = nn.LSTM(input_size, hidden_size)
        
        # LSTMCell for fast sequential evaluation (inference)
        self.cell = nn.LSTMCell(input_size, hidden_size)

        # Sync weights between the training and eval layers
        self.cell.weight_ih = self.lstm.weight_ih_l0
        self.cell.weight_hh = self.lstm.weight_hh_l0
        self.cell.bias_ih = self.lstm.bias_ih_l0
        self.cell.bias_hh = self.lstm.bias_hh_l0

        # Initialize weights
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name and 'layer_norm' not in name:
                nn.init.orthogonal_(param, 1.0)
        

    def forward(self, observations, state):
        '''Forward function for training. Handles state as either tuple (h, c) or dictionary.'''
        B, TT, *space_shape = observations.shape

        encoded_obs = self.policy.encode_observations(
            observations.reshape(B * TT, *space_shape)
        )
        hidden = encoded_obs.reshape(B, TT, self.input_size).transpose(0, 1)
        
        # Handle state as dictionary (from pufferl.py) or tuple
        if isinstance(state, dict):
            h = state.get('lstm_h')
            c = state.get('lstm_c')
            lstm_state = (h, c) if h is not None and c is not None else None
        else:
            lstm_state = state
            
        hidden, (next_h, next_c) = self.lstm.forward(hidden, lstm_state)
        
        flat_hidden = hidden.transpose(0, 1).reshape(B * TT, self.hidden_size)
        
        # Convert sparse legal masks to dense format (chess-specific)
        flat_observations = observations.reshape(B * TT, *space_shape)
        if flat_observations.shape[-1] == 1537:  # Chess sparse format
            # Fixed import
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), 'ocean', 'chess'))
            import sparse_utils
            sparse_to_dense_gpu = sparse_utils.sparse_to_dense_gpu
            legal_masks = sparse_to_dense_gpu(flat_observations)
        else:
            legal_masks = None
        
        logits, values = self.policy.decode_actions(flat_hidden, legal_masks)

        values = values.reshape(B, TT)
        
        # Return only logits and values for training (when lstm_state is None)
        # Return state tuple only when actually using RNN state
        if lstm_state is None:
            return logits, values
        else:
            return logits, values, (next_h.detach(), next_c.detach())

    def forward_eval(self, observations, state):
        '''Forward function for inference. Assumes state is a dictionary.'''
        hidden = self.policy.encode_observations(observations)
        
        # Correctly get LSTM state from the state DICTIONARY
        h = state.get('lstm_h')
        c = state.get('lstm_c')
        
        # Handle LSTM state dimensions - cell expects 2D tensors (batch, hidden)
        if h is not None and h.dim() == 3:
            h = h.squeeze(0)  # Remove layer dimension: (1, batch, hidden) -> (batch, hidden)
        if c is not None and c.dim() == 3:
            c = c.squeeze(0)  # Remove layer dimension: (1, batch, hidden) -> (batch, hidden)
            
        lstm_state = (h, c) if h is not None and c is not None else None

        if lstm_state is not None:
            next_h, next_c = self.cell(hidden, lstm_state)
        else:
            # Initialize LSTM state if not provided
            batch_size = hidden.shape[0]
            device = hidden.device
            next_h = torch.zeros(batch_size, self.hidden_size, device=device)
            next_c = torch.zeros(batch_size, self.hidden_size, device=device)

        # Convert sparse legal masks to dense format (chess-specific)
        if observations.shape[-1] == 1537:  # Chess sparse format
            # Fixed import
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), 'ocean', 'chess'))
            import sparse_utils
            sparse_to_dense_gpu = sparse_utils.sparse_to_dense_gpu
            legal_masks = sparse_to_dense_gpu(observations)
        else:
            legal_masks = None
        
        logits, values = self.policy.decode_actions(next_h, legal_masks)
        
        # Update the state dictionary in-place for the next evaluation step
        # Add back the layer dimension for consistency: (batch, hidden) -> (1, batch, hidden)
        state['lstm_h'] = next_h.unsqueeze(0)
        state['lstm_c'] = next_c.unsqueeze(0)

        return logits, values



    # def forward_eval(self, observations, state):
    #     '''Forward function for inference. 3x faster than using LSTM directly'''
    #     hidden = self.policy.encode_observations(observations, state=state)
    #     h = state['lstm_h']
    #     c = state['lstm_c']

    #     # TODO: Don't break compile
    #     if h is not None:
    #         assert h.shape[0] == c.shape[0] == observations.shape[0], 'LSTM state must be (h, c)'
    #         lstm_state = (h, c)
    #     else:
    #         lstm_state = None

    #     #hidden = self.pre_layernorm(hidden)
    #     hidden, c = self.cell(hidden, lstm_state)
    #     #hidden = self.post_layernorm(hidden)
    #     state['hidden'] = hidden
    #     state['lstm_h'] = hidden
    #     state['lstm_c'] = c
    #     logits, values = self.policy.decode_actions(hidden)
    #     return logits, values

    # def forward(self, observations, state):
    #     '''Forward function for training. Uses LSTM for fast time-batching'''
    #     x = observations
    #     lstm_h = state['lstm_h']
    #     lstm_c = state['lstm_c']

    #     x_shape, space_shape = x.shape, self.obs_shape
    #     x_n, space_n = len(x_shape), len(space_shape)
    #     if x_shape[-space_n:] != space_shape:
    #         raise ValueError('Invalid input tensor shape', x.shape)

    #     if x_n == space_n + 1:
    #         B, TT = x_shape[0], 1
    #     elif x_n == space_n + 2:
    #         B, TT = x_shape[:2]
    #     else:
    #         raise ValueError('Invalid input tensor shape', x.shape)

    #     if lstm_h is not None:
    #         assert lstm_h.shape[1] == lstm_c.shape[1] == B, 'LSTM state must be (h, c)'
    #         lstm_state = (lstm_h, lstm_c)
    #     else:
    #         lstm_state = None

    #     x = x.reshape(B*TT, *space_shape)
    #     hidden = self.policy.encode_observations(x, state)
    #     assert hidden.shape == (B*TT, self.input_size)

    #     hidden = hidden.reshape(B, TT, self.input_size)

    #     hidden = hidden.transpose(0, 1)
    #     #hidden = self.pre_layernorm(hidden)
    #     hidden, (lstm_h, lstm_c) = self.lstm.forward(hidden, lstm_state)
    #     #hidden = self.post_layernorm(hidden)
    #     hidden = hidden.transpose(0, 1)

    #     flat_hidden = hidden.reshape(B*TT, self.hidden_size)
    #     logits, values = self.policy.decode_actions(flat_hidden)
    #     values = values.reshape(B, TT)
    #     #state.batch_logits = logits.reshape(B, TT, -1)
    #     state['hidden'] = hidden
    #     state['lstm_h'] = lstm_h.detach()
    #     state['lstm_c'] = lstm_c.detach()
    #     return logits, values

class Convolutional(nn.Module):
    def __init__(self, env, *args, framestack, flat_size,
            input_size=512, hidden_size=512, output_size=512,
            channels_last=False, downsample=1, **kwargs):
        '''The CleanRL default NatureCNN policy used for Atari.
        It's just a stack of three convolutions followed by a linear layer
        
        Takes framestack as a mandatory keyword argument. Suggested default is 1 frame
        with LSTM or 4 frames without.'''
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample

        #TODO: Remove these from required params
        self.hidden_size = hidden_size
        self.is_continuous = False

        self.network= nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(framestack, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(flat_size, hidden_size)),
            nn.ReLU(),
        )
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(output_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, observations, state=None):
        return self.forward(observations, state)

    def encode_observations(self, observations, state=None):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]
        return self.network(observations.float() / 255.0)

    def decode_actions(self, flat_hidden):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

class ProcgenResnet(nn.Module):
    '''Procgen baseline from the AICrowd NeurIPS 2020 competition
    Based on the ResNet architecture that was used in the Impala paper.'''
    def __init__(self, env, cnn_width=16, mlp_width=256):
        super().__init__()
        h, w, c = env.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [cnn_width, 2*cnn_width, 2*cnn_width]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=mlp_width),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(mlp_width, env.single_action_space.n), std=0.01)
        self.value = pufferlib.pytorch.layer_init(
                nn.Linear(mlp_width, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, observations, state=None):
        return self.forward(observations, state)

    def encode_observations(self, x):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)
        return hidden
 
    def decode_actions(self, hidden):
        '''linear decoder function'''
        action = self.actor(hidden)
        value = self.value(hidden)
        return action, value

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

# Convenience factory so external training scripts can just call
#   pufferlib.models.policy_for(env)
# without caring which concrete implementation is used.

def policy_for(env, **kwargs):
    """
    Policy factory, now hardcoded to always return the ChessRecurrent policy
    wrapped in an LSTMWrapper for your use case.
    """
    from pufferlib.ocean.torch import ChessRecurrent

    hidden = kwargs.pop('hidden_size', 256)
    base = ChessRecurrent(env, hidden_size=hidden)
    # The LSTMWrapper now requires input_size for the hidden state from the base policy
    return LSTMWrapper(env, base, input_size=hidden, hidden_size=hidden)

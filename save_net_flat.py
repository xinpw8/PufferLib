import os
import torch
from torch.nn import functional as F
import numpy as np

# update manually if running manually, otherwise, run.py updates automatically
MODEL_FILE_NAME = '/puffertank/pufferlib/experiments/puffer_enduro-6264e9e3/model_000287.pt'
WEIGHTS_OUTPUT_FILE_NAME = 'enduro_weights.bin'
OUTPUT_FILE_PATH = '/puffertank/pufferlib/pufferlib/resources/enduro'

def detect_action_space_from_model(model):
    """
    Detects whether the model uses continuous or discrete actions.
    Returns a tuple: (action_space_type, action_space_size)
    - action_space_type: 'continuous' or 'discrete' (or 'multidiscrete' if applicable)
    - action_space_size: number of actions (or list in case of multidiscrete)
    """
    # If the model is wrapped (e.g. in an LSTM wrapper), get the underlying policy.
    if hasattr(model, 'policy'):
        policy = model.policy
    else:
        policy = model

    if hasattr(policy, 'decoder_mean') and hasattr(policy, 'decoder_logstd'):
        action_space_type = 'continuous'
        action_space_size = policy.decoder_mean.out_features
    elif hasattr(policy, 'decoder'):
        # Check if decoder is a ModuleList (multidiscrete) or a single module (discrete)
        if isinstance(policy.decoder, torch.nn.ModuleList):
            action_space_type = 'multidiscrete'
            action_space_size = [dec.out_features for dec in policy.decoder]
        else:
            action_space_type = 'discrete'
            action_space_size = policy.decoder.out_features
    else:
        raise ValueError("Cannot determine action space type from model.")
    return action_space_type, action_space_size

def save_model_weights(model, filename):
    """
    Extracts weights from the model and saves them (and the architecture info) to disk.
    Automatically detects whether the policy uses continuous or discrete actions.
    Returns a tuple: (num_weights, action_size, is_continuous)
    """
    weights_path = os.path.join(OUTPUT_FILE_PATH, filename)
    architecture_path = os.path.join(OUTPUT_FILE_PATH, filename + "_architecture.txt")
    os.makedirs(OUTPUT_FILE_PATH, exist_ok=True)
    
    # Detect the action space type and size.
    action_space_type, action_space_size = detect_action_space_from_model(model)
    is_continuous = (action_space_type == 'continuous')
    
    weights = []
    
    # If the model is wrapped (e.g. by an LSTM wrapper)
    if hasattr(model, "policy"):
        policy = model.policy
        
        # Handle Sequential encoder
        if isinstance(policy.encoder, torch.nn.Sequential):
            for layer in policy.encoder:
                if hasattr(layer, 'weight'):
                    weights.append(layer.weight.data.cpu().numpy().flatten())
                if hasattr(layer, 'bias') and layer.bias is not None:
                    weights.append(layer.bias.data.cpu().numpy().flatten())
        else:
            weights.append(policy.encoder.weight.data.cpu().numpy().flatten())
            weights.append(policy.encoder.bias.data.cpu().numpy().flatten())
        
        # Recurrent (for LSTM wrappers)
        if hasattr(model, 'recurrent'):
            weights.append(model.recurrent.weight_ih_l0.data.cpu().numpy().flatten())
            weights.append(model.recurrent.weight_hh_l0.data.cpu().numpy().flatten())
            weights.append(model.recurrent.bias_ih_l0.data.cpu().numpy().flatten())
            weights.append(model.recurrent.bias_hh_l0.data.cpu().numpy().flatten())
        
        if is_continuous:
            weights.append(policy.decoder_mean.weight.data.cpu().numpy().flatten())
            weights.append(policy.decoder_mean.bias.data.cpu().numpy().flatten())
            if hasattr(policy, 'decoder_logstd'):
                weights.append(policy.decoder_logstd.data.cpu().numpy().flatten())
        else:
            if isinstance(policy.decoder, torch.nn.Sequential):
                for layer in policy.decoder:
                    if hasattr(layer, 'weight'):
                        weights.append(layer.weight.data.cpu().numpy().flatten())
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        weights.append(layer.bias.data.cpu().numpy().flatten())
            else:
                weights.append(policy.decoder.weight.data.cpu().numpy().flatten())
                weights.append(policy.decoder.bias.data.cpu().numpy().flatten())
        
        # Value function weights
        if hasattr(policy, 'value'):
            if isinstance(policy.value, torch.nn.Sequential):
                for layer in policy.value:
                    if hasattr(layer, 'weight'):
                        weights.append(layer.weight.data.cpu().numpy().flatten())
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        weights.append(layer.bias.data.cpu().numpy().flatten())
            else:
                weights.append(policy.value.weight.data.cpu().numpy().flatten())
                weights.append(policy.value.bias.data.cpu().numpy().flatten())
        elif hasattr(policy, 'value_mean'):
            weights.append(policy.value_mean.weight.data.cpu().numpy().flatten())
            weights.append(policy.value_mean.bias.data.cpu().numpy().flatten())
            weights.append(policy.value_logstd.data.cpu().numpy().flatten())
    else:
        # Model without an explicit policy attribute (Default model)
        if isinstance(model.encoder, torch.nn.Sequential):
            for layer in model.encoder:
                if hasattr(layer, 'weight'):
                    weights.append(layer.weight.data.cpu().numpy().flatten())
                if hasattr(layer, 'bias') and layer.bias is not None:
                    weights.append(layer.bias.data.cpu().numpy().flatten())
        else:
            weights.append(model.encoder.weight.data.cpu().numpy().flatten())
            weights.append(model.encoder.bias.data.cpu().numpy().flatten())
            
        if is_continuous:
            weights.append(model.decoder_mean.weight.data.cpu().numpy().flatten())
            weights.append(model.decoder_mean.bias.data.cpu().numpy().flatten())
            if hasattr(model, 'decoder_logstd'):
                weights.append(model.decoder_logstd.data.cpu().numpy().flatten())
        else:
            if isinstance(model.decoder, torch.nn.Sequential):
                for layer in model.decoder:
                    if hasattr(layer, 'weight'):
                        weights.append(layer.weight.data.cpu().numpy().flatten())
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        weights.append(layer.bias.data.cpu().numpy().flatten())
            else:
                weights.append(model.decoder.weight.data.cpu().numpy().flatten())
                weights.append(model.decoder.bias.data.cpu().numpy().flatten())
                
        if hasattr(model, 'value'):
            if isinstance(model.value, torch.nn.Sequential):
                for layer in model.value:
                    if hasattr(layer, 'weight'):
                        weights.append(layer.weight.data.cpu().numpy().flatten())
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        weights.append(layer.bias.data.cpu().numpy().flatten())
            else:
                weights.append(model.value.weight.data.cpu().numpy().flatten())
                weights.append(model.value.bias.data.cpu().numpy().flatten())
        elif hasattr(model, 'value_mean'):
            weights.append(model.value_mean.weight.data.cpu().numpy().flatten())
            weights.append(model.value_mean.bias.data.cpu().numpy().flatten())
            weights.append(model.value_logstd.data.cpu().numpy().flatten())
    
    weights = np.concatenate(weights)
    weights.tofile(weights_path)
    
    # Write architecture info to a text file.
    with open(architecture_path, "w") as f:
        f.write("Model Architecture:\n")
        if hasattr(model, "policy"):
            policy = model.policy
            f.write("\nEncoder:\n")
            if isinstance(policy.encoder, torch.nn.Sequential):
                for i, layer in enumerate(policy.encoder):
                    f.write(f"Layer {i}: {layer}\n")
                    if hasattr(layer, 'weight'):
                        f.write(f"  weight: {layer.weight.shape}\n")
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        f.write(f"  bias: {layer.bias.shape}\n")
            else:
                f.write(f"encoder.weight: {policy.encoder.weight.shape}\n")
                f.write(f"encoder.bias: {policy.encoder.bias.shape}\n")
        
        if is_continuous:
            if hasattr(model, "policy"):
                action_size = policy.decoder_mean.weight.shape[0]
                f.write(f"\nDecoder Mean:\n")
                f.write(f"decoder_mean.weight: {policy.decoder_mean.weight.shape}\n")
                f.write(f"decoder_mean.bias: {policy.decoder_mean.bias.shape}\n")
                if hasattr(policy, 'decoder_logstd'):
                    f.write(f"decoder_logstd: {policy.decoder_logstd.shape}\n")
                    f.write(f"decoder_logstd_values: {','.join(map(str, policy.decoder_logstd.data.cpu().numpy()))}\n")
            else:
                action_size = model.decoder_mean.weight.shape[0]
            f.write(f"\nAction size: {action_size}\n")
            f.write(f"Action type: continuous\n")
        else:
            if hasattr(model, "policy"):
                if isinstance(policy.decoder, torch.nn.Sequential):
                    action_size = policy.decoder[-1].weight.shape[0]  # Get output size of last layer
                else:
                    action_size = policy.decoder.weight.shape[0]
            else:
                if isinstance(model.decoder, torch.nn.Sequential):
                    action_size = model.decoder[-1].weight.shape[0]
                else:
                    action_size = model.decoder.weight.shape[0]
            f.write(f"\nAction size: {action_size}\n")
            f.write(f"Action type: discrete\n")
        f.write(f"Num weights: {len(weights)}\n")
    
    print(f"Saved model weights to {weights_path} and architecture to {architecture_path}")
    return len(weights), action_size, is_continuous
        
def test_model(model):
    model = model.cpu().policy
    batch_size = 16
    obs_window = 11
    obs_window_channels = 4
    obs_flat = 26
    x = torch.arange(
        0, batch_size*(obs_window*obs_window*obs_window_channels + obs_flat)
        ).reshape(batch_size, -1) % 16

    cnn_features = x[:, :-obs_flat].view(
        batch_size, obs_window, obs_window, obs_window_channels).long()
    map_features = F.one_hot(cnn_features[:, :, :, 0], 16).permute(0, 3, 1, 2).float()
    extra_map_features = (cnn_features[:, :, :, -3:].float() / 255.0).permute(0, 3, 1, 2)
    cnn_features = torch.cat([map_features, extra_map_features], dim=1)
    cnn = model.policy.cnn

    cnn_features = torch.from_numpy(
        np.arange(batch_size*11*11*19).reshape(
        batch_size, 19, obs_window, obs_window)
    ).float()
    conv1_out = cnn[0](cnn_features)

    #(cnn[0].weight[0] * cnn_features[0, :, :5, :5]).sum() + cnn[0].bias[0]

    breakpoint()
    hidden = model.encoder(x)
    output = model.decoder(hidden)
    atn = output.argmax(dim=1)
    print('Encode weight sum:', model.encoder.weight.sum())
    print('encode decode weight and bias sum:', model.encoder.weight.sum() + model.encoder.bias.sum() + model.decoder.weight.sum() + model.decoder.bias.sum())
    print('X sum:', x.sum())
    print('Hidden sum:', hidden.sum())
    print('Hidden 1-10:', hidden[0, :10])
    print('Output sum:', output.sum())
    print('Atn sum:', atn.sum())
    breakpoint()
    exit(0)

def test_lstm():
    batch_size = 16
    input_size = 128
    hidden_size = 128

    input = torch.arange(batch_size*input_size).reshape(1, batch_size, -1).float()/ 100000
    state = (
        torch.arange(batch_size*hidden_size).reshape(1, batch_size, -1).float()/ 100000,
        torch.arange(batch_size*hidden_size).reshape(1, batch_size, -1).float() / 100000
    )
    weights_input = torch.arange(4*hidden_size*input_size).reshape(4*hidden_size, -1).float()/ 100000
    weights_state = torch.arange(4*hidden_size*hidden_size).reshape(4*hidden_size, -1).float()/ 100000
    bias_input = torch.arange(4*hidden_size).reshape(4*hidden_size).float() / 100000
    bias_state = torch.arange(4*hidden_size).reshape(4*hidden_size).float() / 100000

    lstm = torch.nn.LSTM(input_size=128, hidden_size=128, num_layers=1)
    lstm.weight_ih_l0.data = weights_input
    lstm.weight_hh_l0.data = weights_state
    lstm.bias_ih_l0.data = bias_input
    lstm.bias_hh_l0.data = bias_state

    output, new_state = lstm(input, state)

    input = input.squeeze(0)
    h, c = state

    buffer = (
        torch.matmul(input, weights_input.T) + bias_input
        + torch.matmul(h, weights_state.T) + bias_state
    )[0]

    i, f, g, o = torch.split(buffer, hidden_size, dim=1)

    i = torch.sigmoid(i)
    f = torch.sigmoid(f)
    g = torch.tanh(g)
    o = torch.sigmoid(o)

    c = f*c + i*g
    h = o*torch.tanh(c)

    breakpoint()
    print('Output:', output)

def test_model_forward(model):
    data = torch.arange(10*(11*11*4 + 26)) % 16
    data[(11*11*4 + 26):] = 0
    data = data.reshape(10, -1).float()
    output = model(data)
    breakpoint()
    pass

	
if __name__ == '__main__':
    #test_lstm()
    model = torch.load(MODEL_FILE_NAME, map_location='cpu', weights_only=False)
    print(f"loaded weights from {MODEL_FILE_NAME}")
    #test_model_forward(model)
    #test_model(model)

    save_model_weights(model, WEIGHTS_OUTPUT_FILE_NAME)
    print('saved')

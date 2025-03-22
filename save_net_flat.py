import torch
from torch.nn import functional as F
import numpy as np

# update manually if running manually, otherwise, run.py updates automatically
MODEL_FILE_NAME = '/puffertank/pufferlib/experiments/puffer_breakout-e033b016/model_000382.pt'
WEIGHTS_OUTPUT_FILE_NAME = 'breakout_weights.bin'
OUTPUT_FILE_PATH = '/puffertank/pufferlib/pufferlib/resources/breakout'

def save_model_weights(model, filename):
    import os
    weights_path = os.path.join(OUTPUT_FILE_PATH, filename)
    architecture_path = os.path.join(OUTPUT_FILE_PATH, filename + "_architecture.txt")
    
    os.makedirs(OUTPUT_FILE_PATH, exist_ok=True)
    
    weights = []

    # encoder first
    weights.append(model.policy.encoder.weight.data.cpu().numpy().flatten())
    weights.append(model.policy.encoder.bias.data.cpu().numpy().flatten())

    # LSTM next
    weights.append(model.recurrent.weight_ih_l0.data.cpu().numpy().flatten())
    weights.append(model.recurrent.weight_hh_l0.data.cpu().numpy().flatten())
    weights.append(model.recurrent.bias_ih_l0.data.cpu().numpy().flatten())
    weights.append(model.recurrent.bias_hh_l0.data.cpu().numpy().flatten())

    # decoder_mean after LSTM
    weights.append(model.policy.decoder_mean.weight.data.cpu().numpy().flatten())
    weights.append(model.policy.decoder_mean.bias.data.cpu().numpy().flatten())

    # explicitly save decoder_logstd separately or at end
    weights.append(model.policy.decoder_logstd.data.cpu().numpy().flatten())

    weights = np.concatenate(weights)
    weights.tofile(weights_path)
    
    # weights = []
    # for name, param in model.named_parameters():
    #     weights.append(param.data.cpu().numpy().flatten())
    #     print(name, param.shape, param.data.cpu().numpy().ravel()[0])
    
    # weights = np.concatenate(weights)
    # print('Num weights:', len(weights))
    # weights.tofile(weights_path)
    
    # Extract action dimensions from the model architecture
    action_size = 0
    action_is_continuous = False
    
    # Identify action dimension from decoder weights
    for name, param in model.named_parameters():
        if 'decoder_mean.weight' in name:
            action_size = param.shape[0]
            action_is_continuous = True
            print(f"Detected continuous action space with size: {action_size}")
            break
        elif 'decoder.weight' in name and not action_size:
            action_size = param.shape[0]
            print(f"Detected discrete action space with size: {action_size}")
    
    # Write out full architecture info along with action details in one go
    with open(architecture_path, "w") as f:
        for name, param in model.named_parameters():
            f.write(f"{name}: {param.shape}\n")

        # Explicitly save numeric decoder_logstd values
        decoder_logstd_tensor = model.policy.decoder_logstd.detach().cpu().numpy().flatten()
        decoder_logstd_str = ', '.join(str(x) for x in decoder_logstd_tensor)
        f.write(f"decoder_logstd_values: {decoder_logstd_str}\n")

        f.write(f"Num weights: {len(weights)}\n")
        f.write(f"Action size: {action_size}\n")
        f.write(f"Action type: {'continuous' if action_is_continuous else 'discrete'}\n")

    print(f"Saved model weights to {weights_path} and architecture to {architecture_path}")
    
    # Return detected values for use by calling functions
    return len(weights), action_size, action_is_continuous
        
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
    model = torch.load(MODEL_FILE_NAME, map_location='cpu')
    print(f"loaded weights from {MODEL_FILE_NAME}")
    #test_model_forward(model)
    #test_model(model)

    save_model_weights(model, WEIGHTS_OUTPUT_FILE_NAME)
    print('saved')

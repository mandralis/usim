import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from IPython import embed
from learning.deprecated.train_convlstm import ConvLSTMNet

# Load data
train_data           = torch.load('train_data.pth')
val_data             = torch.load('val_data.pth')
X_train, Theta_train = train_data['X'], train_data['Theta']
X_val, Theta_val     = val_data['X'], val_data['Theta']

# dimensions
dim_n_joints     = 8
seq_length       = 4
lstm_hidden_size = 32
lstm_num_layers  = 1
layer_dims       = [256,128,64,32]
dim_in           = X_train.shape[1]
dim_out          = Theta_train.shape[1]

# Load the model
model_path = "/Users/imandralis/src/usim/src/learning/learned_models/20240620-111753/model_interrupted.pth"
model = ConvLSTMNet(dim_in, dim_out, seq_length, lstm_hidden_size, lstm_num_layers, layer_dims)
model.load_state_dict(torch.load(model_path))
model.eval()

# also save as onnx for use in Matlab
import torch.onnx
example_input = X_train[0].unsqueeze(0)
torch.onnx.export(model, example_input, "model.onnx", export_params=True, opset_version=11,
                do_constant_folding=True, input_names=['input'], output_names=['output'])

# Perform inference
Theta_predicted = model(X_val)
print(Theta_val)
plt.figure(figsize=(12, 6))
for i in range(dim_out):
    plt.subplot(3, 3, i+1)
    plt.plot(Theta_val[:, i].detach().numpy(), 'b', label='Actual')
    plt.plot(Theta_predicted[:, i].detach().numpy(), 'r', label='Predicted')
    plt.xlabel('Sample')
    plt.ylabel('Angle')
    plt.title(f'Angle {i+1}')
    plt.legend()
plt.tight_layout()
plt.show()
embed()

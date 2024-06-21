import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from train_convlstm import ConvLSTMNet, TrainingDataset, DataLoader
from torch.utils.data import Dataset, DataLoader

# Train folder path
train_path = "/Users/imandralis/src/usim/src/learning/learned_models/20240620-141539"

# Load data
train_data           = torch.load(os.path.join(train_path, 'train_data.pth'))
val_data             = torch.load(os.path.join(train_path, 'val_data.pth'))
X_train, Theta_train = train_data['X'].cpu(), train_data['Theta'].cpu()
X_val, Theta_val     = val_data['X'].cpu(), val_data['Theta'].cpu()
X = torch.cat((X_train, X_val))
Theta = torch.cat((Theta_train, Theta_val))

# Load config
config               = torch.load(os.path.join(train_path, 'config.pth'))

# create train loader
train_dataset = TrainingDataset(X_train, Theta_train, config.get('seq_length'))
train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size'), shuffle=False)

# create val loader
val_dataset = TrainingDataset(X_val, Theta_val, config.get('seq_length'))
val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size'), shuffle=False)

# Load the model
model = ConvLSTMNet(config.get('dim_in'),config.get('dim_out'),config.get('lstm_hidden_size'),config.get('lstm_num_layers'),config.get('layer_dims'))
model.load_state_dict(torch.load(os.path.join(train_path, 'model.pth')))
model.eval()

# # also save as onnx for use in Matlab
# import torch.onnx

# # get one element from train loader
# X_train, Theta_train = next(iter(train_loader))
# torch.onnx.export(model, X_train, "model.onnx", export_params=True, opset_version=11,
#                 do_constant_folding=True, input_names=['input'], output_names=['output'])

# Get predicted angles on validation set using validation loader
with torch.no_grad():
    Theta_predicted_val = []
    for X_val_, Theta_val_ in val_loader:
        output = model(X_val_)
        Theta_predicted_val.append(output.squeeze())
    Theta_predicted_val = torch.cat(Theta_predicted_val)

# Get predicted angles on train set using train loader
with torch.no_grad():
    Theta_predicted_train = []
    for X_train_, Theta_train_ in train_loader:
        output = model(X_train_)
        Theta_predicted_train.append(output.squeeze())
    Theta_predicted_train = torch.cat(Theta_predicted_train)

# # reshape predictions for plotting
# Theta_predicted_val = Theta_predicted_val.view(-1, config.get('dim_out'))
# Theta_predicted_train = Theta_predicted_train.view(-1, config.get('dim_out'))

# unnormalize the validation angles
Theta_mean = val_data['Theta_mean'].cpu()
Theta_std = val_data['Theta_std'].cpu()
Theta_predicted_val = Theta_predicted_val * Theta_std + Theta_mean
Theta_val = Theta_val * Theta_std + Theta_mean

# unnormalize the training angles
Theta_mean = train_data['Theta_mean'].cpu()
Theta_std = train_data['Theta_std'].cpu()
Theta_predicted_train = Theta_predicted_train * Theta_std + Theta_mean
Theta_train = Theta_train * Theta_std + Theta_mean

embed()

# Plot the results on the validation set
plt.figure(figsize=(12, 6))
for i in range(config.get('dim_out')):
    plt.subplot(3, 3, i+1)
    plt.plot(Theta_val[:, i].detach().numpy(), 'b', label='Actual')
    plt.plot(Theta_predicted_val[:, i].detach().numpy(), 'r', label='Predicted')
    plt.xlabel('Sample')
    plt.ylabel('Angle')
    plt.title(f'Angle {i+1}')
    plt.legend()
plt.tight_layout()

# Plot the results on the training set
plt.figure(figsize=(12, 6))
for i in range(config.get('dim_out')):
    plt.subplot(3, 3, i+1)
    plt.plot(Theta_train[:, i].detach().numpy(), 'b', label='Actual')
    plt.plot(Theta_predicted_train[:, i].detach().numpy(), 'r', label='Predicted')
    plt.xlabel('Sample')
    plt.ylabel('Angle')
    plt.title(f'Angle {i+1}')
    plt.legend()
plt.tight_layout()

plt.show()
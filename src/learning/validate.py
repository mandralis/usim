import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from train import ConvNet

# Train folder path
train_path = "/Users/imandralis/src/usim/src/learning/learned_models/best_conv1d"

# Load data
train_data           = torch.load(os.path.join(train_path, 'train_data.pth'))
val_data             = torch.load(os.path.join(train_path, 'val_data.pth'))
# X_train, Theta_train = train_data['X'].cpu(), train_data['Theta'].cpu()
# X_val, Theta_val     = val_data['X'].cpu(), val_data['Theta'].cpu()
# X = torch.cat((X_train, X_val))
# Theta = torch.cat((Theta_train, Theta_val))
X = train_data['X'].cpu()
Theta = train_data['Theta'].cpu()

# Load config
config               = torch.load(os.path.join(train_path, 'config.pth'))

# Load the model
model = ConvNet(config.get('dim_in'),config.get('dim_out'),config.get('layer_dims'))
model.load_state_dict(torch.load(os.path.join(train_path, 'model.pth')))
model.eval()

# also save as onnx for use in Matlab
import torch.onnx
example_input = X[0].unsqueeze(0)
torch.onnx.export(model, example_input, "model.onnx", export_params=True, opset_version=11,
                do_constant_folding=True, input_names=['input'], output_names=['output'])

# Perform inference
Theta_predicted_val = model(X)

# unnormalize the angles
Theta_mean = val_data['Theta_mean'].cpu()
Theta_std = val_data['Theta_std'].cpu()
Theta_predicted_val = Theta_predicted_val * Theta_std + Theta_mean
Theta_val = Theta * Theta_std + Theta_mean

# Plot the results on the full set (training + validation)
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
plt.show()
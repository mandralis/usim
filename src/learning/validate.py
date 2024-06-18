import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from IPython import embed
from train import ConvNet, TrainingDataset

# load data
X     = sio.loadmat('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/X.mat')['X'][:,:2000]
Theta = sio.loadmat('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/Theta_relative_8_joints.mat')['Theta_relative']

# Convert data to tensor and float32
X = torch.tensor(X).float()
Theta = torch.tensor(Theta).float()

# Normalize the input data and output data
X_mean = torch.mean(X, dim=0)
X_std = torch.std(X, dim=0)
X = (X - X_mean) / (X_std+1e-6)

Theta_mean = torch.mean(Theta, dim=0)
Theta_std = torch.std(Theta, dim=0)
Theta = (Theta - Theta_mean) / (Theta_std+1e-6)

# Print means and stds
print("X_mean:", X_mean)
print("X_std:", X_std)
print("Theta_mean:", Theta_mean)
print("Theta_std:", Theta_std)

# Convert data to float32
X = torch.tensor(X).float()
Theta = torch.tensor(Theta).float()
# train_dataset = TrainingDataset(X, Theta)

# # also load saved data
# train_data = torch.load('train_data.pth')    
# val_data = torch.load('val_data.pth')    
# X_train,Theta_train = train_data["X"], train_data["Theta"]
# X_val,Theta_val = val_data["X"], val_data["Theta"]

# dimensions
dim_in        = X.shape[1]
dim_out       = Theta.shape[1]

# Load the model
model_path = "model.pth"
model = ConvNet(dim_in,dim_out)
model.load_state_dict(torch.load(model_path))
model.eval()

# also save as onnx for use in Matlab
import torch.onnx
example_input = X[0].unsqueeze(0)
torch.onnx.export(model, example_input, "model.onnx", export_params=True, opset_version=11,
                do_constant_folding=True, input_names=['input'], output_names=['output'])

# Perform inference
batch_size = 3990  # all data including train and validation data 
Theta_predicted = model(X[:batch_size,:])
print(Theta_predicted)
print(Theta[:batch_size,:])
plt.figure(figsize=(12, 6))
for i in range(dim_out):
    plt.subplot(3, 3, i+1)
    plt.plot(Theta[:batch_size, i].detach().numpy(), 'b', label='Actual')
    plt.plot(Theta_predicted[:batch_size, i].detach().numpy(), 'r', label='Predicted')
    plt.xlabel('Sample')
    plt.ylabel('Angle')
    plt.title(f'Angle {i+1}')
    plt.legend()
plt.tight_layout()
plt.show()
embed()

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from IPython import embed
from train_conv1d import ConvNet, TrainingDataset

# load the data
data = sio.loadmat('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/X.mat')
X = data['X'][:,:2000]

data = sio.loadmat('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/Theta_relative.mat')
Theta = data['Theta_relative']

# Normalize the input data and output data
X_mean = torch.mean(X, dim=0)
X_std = torch.std(X, dim=0)
X = (X - X_mean) / X_std

Theta_mean = torch.mean(Theta, dim=0)
Theta_std = torch.std(Theta, dim=0)
Theta = (Theta - Theta_mean) / Theta_std

# Convert data to float32
X = torch.tensor(X).float()
Theta = torch.tensor(Theta).float()
# train_dataset = TrainingDataset(X, Theta)

# dimensions
dim_in        = X.shape[1]
dim_out       = Theta.shape[1]

# Load the model
model_path = "model.pth"
model = ConvNet(dim_in,dim_out)
model.load_state_dict(torch.load(model_path))
model.eval()

# Perform inference
Theta_predicted = model(X[:100,:])
print(Theta_predicted)
print(Theta[:100,:])

plt.plot(Theta[:100,0].detach().numpy(),'b')
plt.plot(Theta_predicted[:100,0].detach().numpy(),'r')

plt.show()
embed()


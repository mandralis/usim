import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from IPython import embed

# Load the training data
import scipy.io as sio

data = sio.loadmat('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/X.mat')
X = data['X']

data = sio.loadmat('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/Theta_relative.mat')
Theta = data['Theta_relative']

# Normalize the input data
mean_X = np.mean(X, axis=0)
std_X = np.std(X, axis=0)
X_normalized = (X - mean_X) / std_X

# Normalize the output data
mean_Theta = np.mean(Theta, axis=0)
std_Theta = np.std(Theta, axis=0)
Theta_normalized = (Theta - mean_Theta) / std_Theta

# Define the transformer model
input_dim = X_normalized.shape[1]
output_dim = Theta.shape[1]
model = nn.Transformer()

embed()

# # Set up optimizer
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.MSELoss()

# # Convert the data to PyTorch tensors
# X_tensor = torch.from_numpy(X_normalized).float()
# Theta_tensor = torch.from_numpy(Theta).float()

# # Training loop
# num_epochs = 1000
# for epoch in range(num_epochs):
#     # Forward pass
#     outputs = model(X_tensor)
#     loss = criterion(outputs, Theta_tensor)

#     # Backward and optimize
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # Print progress
#     if (epoch+1) % 100 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# # Save the trained model
# torch.save(model.state_dict(), 'trained_model.pth')
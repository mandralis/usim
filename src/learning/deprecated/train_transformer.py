import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np 
from IPython import embed

import torch.nn as nn
import torch.optim as optim
import scipy.io as sio

mps_device = torch.device("mps")

# load data
X     = sio.loadmat('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/X.mat')['X'][:,:2000]
Theta = sio.loadmat('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/Theta_relative_8_joints.mat')['Theta_relative']

# Convert data to tensor and float32
X = torch.tensor(X).float().to(mps_device)
Theta = torch.tensor(Theta).float().to(mps_device)

# Normalize the input data and output data
X_mean = torch.mean(X, dim=0)
X_std = torch.std(X, dim=0) + 1e-6
X = (X - X_mean) / X_std

Theta_mean = torch.mean(Theta, dim=0)
Theta_std = torch.std(Theta, dim=0) + 1e-6
Theta = (Theta - Theta_mean) / Theta_std

# Define custom dataset class
class TrainingDataset(Dataset):
    def __init__(self, X, Theta):
        self.X     = X
        self.Theta = Theta

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        theta = self.Theta[idx]
        return x, theta

# Define transformer neural network model
class TransformerNet(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(TransformerNet, self).__init__()
        self.transformer = nn.Transformer(d_model=dim_in, nhead=1, num_encoder_layers=1, num_decoder_layers=1)
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.T
        x = self.transformer(x.T, x.T)
        x = self.fc(x)
        return x
    
if __name__ == '__main__':
    # Set hyperparameters
    batch_size    = 32
    dim_in        = X.shape[1]
    dim_out       = Theta.shape[1]
    learning_rate = 0.001
    num_epochs    = 300
    validation_split = 0.2  # 20% of the data will be used for validation

    # Create train-validation split
    X_train, X_val, Theta_train, Theta_val = train_test_split(X, Theta, test_size=validation_split)

    # Save train and validation data for future use
    train_data = {
        'X': X_train,
        'Theta': Theta_train,
        'X_mean': X_mean,
        'X_std': X_std,
        'Theta_mean': Theta_mean,
        'Theta_std': Theta_std
    }
    val_data = {
        'X': X_val,
        'Theta': Theta_val,
        'X_mean': X_mean,
        'X_std': X_std,
        'Theta_mean': Theta_mean,
        'Theta_std': Theta_std
    }
    torch.save(train_data, 'train_data.pth')
    torch.save(val_data, 'val_data.pth')

    # Create an instance of the TransformerNet model
    model = TransformerNet(dim_in, dim_out)

    # Convert model parameters to float32
    model = model.float()
    model = model.to(mps_device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create training dataset and data loader
    train_dataset = TrainingDataset(X_train, Theta_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create validation dataset and data loader
    val_dataset = TrainingDataset(X_val, Theta_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        for inputs, targets in train_loader:
            # Forward pass
            inputs = inputs.float()
            targets = targets.float()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.float()
                targets = targets.float()
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        val_loss /= len(val_loader)

        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Validation Loss: {val_loss}")

    # Save the trained model
    torch.save(model.state_dict(), 'model.pth')

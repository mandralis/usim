import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from IPython import embed
import datetime
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio

# Set device
device = torch.device('mps')

# Load data
X = sio.loadmat('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/X.mat')['X'][:, :2000]
Theta = sio.loadmat('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/Theta_relative_8_joints.mat')['Theta_relative']

# Convert data to tensor and float32
X = torch.tensor(X).float().to(device)
Theta = torch.tensor(Theta).float().to(device)

# Normalize the input data and output data
X_mean = torch.mean(X, dim=0)
X_std = torch.std(X, dim=0) + 1e-6
X = (X - X_mean) / X_std

Theta_mean = torch.mean(Theta, dim=0)
Theta_std = torch.std(Theta, dim=0) + 1e-6
Theta = (Theta - Theta_mean) / Theta_std

# Define custom dataset class to handle sequences
class TrainingDataset(Dataset):
    def __init__(self, X, Theta, seq_length):
        self.X = X
        self.Theta = Theta
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx+self.seq_length]
        theta_seq = self.Theta[idx:idx+self.seq_length]
        return x_seq, theta_seq

# Define 1D convolutional neural network model with LSTM and batch normalization
class ConvLSTMNet(nn.Module):
    def __init__(self, dim_in, dim_out, lstm_hidden_size=64, lstm_num_layers=2, layer_dims=[512,256,128,64]):
        super(ConvLSTMNet, self).__init__()
        self.conv1 = nn.Conv1d(dim_in, layer_dims[0], kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm1d(layer_dims[0])
        self.conv2 = nn.Conv1d(layer_dims[0], layer_dims[1], kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm1d(layer_dims[1])
        self.conv3 = nn.Conv1d(layer_dims[1], layer_dims[2], kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm1d(layer_dims[2])
        self.conv4 = nn.Conv1d(layer_dims[2], layer_dims[3], kernel_size=3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm1d(layer_dims[3])
        self.relu  = nn.ReLU()
        self.lstm  = nn.LSTM(input_size=layer_dims[3], hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)
        self.fc    = nn.Linear(lstm_hidden_size, dim_out)

    def forward(self, x):
        # reshape input
        batch_size, seq_length, dim_in = x.shape
        x = x.view(dim_in,batch_size * seq_length)

        # first layer
        x = self.conv1(x)
        x = self.bn1(x.T)
        x = self.relu(x)

        # second layer
        x = self.conv2(x.T)
        x = self.bn2(x.T)
        x = self.relu(x)

        # third layer
        x = self.conv3(x.T)
        x = self.bn3(x.T)
        x = self.relu(x)

        # fourth layer
        x = self.conv4(x.T)
        x = self.bn4(x.T)
        x = self.relu(x)

        # reshape back to batch and sequence dimensions for LSTM
        x = x.view(batch_size, seq_length, -1)
        x, _ = self.lstm(x)
        
        # regression layer
        x = self.fc(x)
        return x

def custom_loss(pred_theta, true_theta):
    mse_loss = nn.MSELoss()
    loss = mse_loss(pred_theta, true_theta)
    return loss

if __name__ == '__main__':
    # Set hyperparameters
    dim_n_joints     = 8
    batch_size       = 32
    seq_length       = 4 
    layer_dims       = [256,128,64,32]
    dim_in           = X.shape[1]
    dim_out          = Theta.shape[1]
    lstm_hidden_size = 32
    lstm_num_layers  = 1
    learning_rate    = 0.001
    num_epochs       = 1500
    validation_split = 0.2 

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
    torch.save(val_data,   'val_data.pth')

    # Create an instance of the ConvLSTMNet model
    model = ConvLSTMNet(dim_in, dim_out, lstm_hidden_size, lstm_num_layers, layer_dims)

    # Convert model parameters to float32 and put on device
    model = model.float().to(device)

    # Define loss function and optimizer
    criterion = custom_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create training dataset and data loader
    train_dataset = TrainingDataset(X_train, Theta_train, seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Create validation dataset and data loader
    val_dataset = TrainingDataset(X_val, Theta_val, seq_length)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Set up TensorBoard
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", timestamp)
    writer = SummaryWriter(log_dir)

    # Training loop
    try:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                # Forward pass
                inputs = inputs.float().to(device)
                targets = targets.float().to(device)
                embed()
                outputs = model(inputs)
                loss = criterion(outputs, targets)  # Use the last time step target for loss

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.float().to(device)
                    targets = targets.float().to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()

            val_loss /= len(val_loader)

            # Log to TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)

            # Print progress
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss}, Validation Loss: {val_loss}")

    except KeyboardInterrupt:
        print('Training interrupted. Saving the model...')
        # Save the model
        folder_path = f"/Users/imandralis/src/usim/src/learning/learned_models/{timestamp}"
        os.makedirs(folder_path, exist_ok=True)
        model_path = os.path.join(folder_path, "model_interrupted.pth")
        torch.save(model.state_dict(), model_path)
        print(f'Model saved at {model_path}')

    # Close the TensorBoard writer
    writer.close()

    # Save the trained model
    folder_path = f"/Users/imandralis/src/usim/src/learning/learned_models/{timestamp}"
    os.makedirs(folder_path, exist_ok=True)

    # Save the trained model
    model_path = os.path.join(folder_path, "model.pth")
    torch.save(model.state_dict(), model_path)

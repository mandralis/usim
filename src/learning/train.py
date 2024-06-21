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

# Define custom dataset class
class TrainingDataset(Dataset):
    def __init__(self, X, Theta):
        self.X = X
        self.Theta = Theta

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        theta = self.Theta[idx]
        return x, theta

# Define 1D convolutional neural network model with batch normalization
class ConvNet(nn.Module):
    def __init__(self, dim_in, dim_out, layer_dims=[512,256,128,64], kernel_size=3, stride=1, padding=1):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(dim_in, layer_dims[0], kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(layer_dims[0])
        self.conv2 = nn.Conv1d(layer_dims[0], layer_dims[1], kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(layer_dims[1])
        self.conv3 = nn.Conv1d(layer_dims[1], layer_dims[2], kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3 = nn.BatchNorm1d(layer_dims[2])
        self.conv4 = nn.Conv1d(layer_dims[2], layer_dims[3], kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn4 = nn.BatchNorm1d(layer_dims[3])
        self.relu = nn.ReLU()
        self.fc = nn.Linear(layer_dims[3] , dim_out)

    def forward(self, x_):
        x = x_

        # pass everything into network
        x = x.T
        x = self.conv1(x)
        x = self.bn1(x.T)
        x = self.relu(x)
        x = self.conv2(x.T)
        x = self.bn2(x.T)
        x = self.relu(x)
        x = self.conv3(x.T)
        x = self.bn3(x.T)
        x = self.relu(x)
        x = self.conv4(x.T)
        x = self.bn4(x.T)
        x = self.relu(x)
        
        # concatenate 
        x = self.fc(x)
        return x

def custom_loss(pred_theta, true_theta):
    mse_loss = nn.MSELoss()
    loss1 = mse_loss(pred_theta, true_theta)
    return loss1

if __name__ == '__main__':
    # Set device
    device = torch.device('mps')

    # Load data
    nx_start, nx_end = 1200, 2200
    X     = sio.loadmat('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/X.mat')['X'][:, nx_start:nx_end]
    Theta = sio.loadmat('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/Theta_relative_8_joints.mat')['Theta_relative']

    # Convert data to tensor and float32
    X = torch.tensor(X).float().to(device)
    Theta = torch.tensor(Theta).float().to(device)

    # Normalize the input data
    X_mean = torch.mean(X, dim=0)
    X_std = torch.std(X, dim=0) + 1e-6
    X = (X - X_mean) / X_std

    # Normalize the output data
    Theta_mean = torch.mean(Theta, dim=0)
    Theta_std = torch.std(Theta, dim=0) + 1e-6
    Theta = (Theta - Theta_mean) / Theta_std

    # Set hyperparameters
    dim_n_joints = 8
    batch_size = 128
    layer_dims = [512,256,128,64]
    kernel_size = 3
    dim_in = X.shape[1]
    dim_out = Theta.shape[1]
    learning_rate = 0.001
    num_epochs = 1500
    validation_split = 0.2  # 20% of the data will be used for validation

    # Create train folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_path = f"/Users/imandralis/src/usim/src/learning/learned_models/{timestamp}"
    os.makedirs(folder_path, exist_ok=True)

    # Save config
    config = {
        'dim_n_joints': dim_n_joints,
        'batch_size': batch_size,
        'layer_dims': layer_dims,
        'kernel_size': kernel_size,
        'dim_in': dim_in,
        'dim_out': dim_out,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'validation_split': validation_split
    }
    config_path = os.path.join(folder_path, "config.pth")
    torch.save(config, config_path)
    print(f'Config saved at {config_path}')

    # Create train-validation split
    X_train, X_val, Theta_train, Theta_val = train_test_split(X, Theta, test_size=validation_split, shuffle=True)

    # Save train and validation data for future use
    train_data = {
        'X': X,
        'Theta': Theta,
        'X_train': X_train,
        'Theta_train': Theta_train,
        'X_mean': X_mean,
        'X_std': X_std,
        'Theta_mean': Theta_mean,
        'Theta_std': Theta_std
    }
    val_data = {
        'X': X,
        'Theta': Theta,
        'X_val': X_val,
        'Theta_val': Theta_val,
        'X_mean': X_mean,
        'X_std': X_std,
        'Theta_mean': Theta_mean,
        'Theta_std': Theta_std
    }
    train_data_path = os.path.join(folder_path, "train_data.pth")
    val_data_path = os.path.join(folder_path, "val_data.pth")
    torch.save(train_data, train_data_path)
    torch.save(val_data, val_data_path)
    print(f'Train data saved at {train_data_path}')
    print(f'Validation data saved at {val_data_path}')
    
    # Create an instance of the ConvNet model
    model = ConvNet(dim_in, dim_out, layer_dims, kernel_size=kernel_size)

    # Convert model parameters to float32 and put on device
    model = model.float().to(device)

    # Define loss function and optimizer
    criterion = custom_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create training dataset and data loader
    train_dataset = TrainingDataset(X_train, Theta_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create validation dataset and data loader
    val_dataset = TrainingDataset(X_val, Theta_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Set up TensorBoard
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
                inputs = inputs.float()
                targets = targets.float()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

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
                    inputs = inputs.float()
                    targets = targets.float()
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
        model_path = os.path.join(folder_path, "model_interrupted.pth")
        torch.save(model.state_dict(), model_path)
        print(f'Model saved at {model_path}')

    # Close the TensorBoard writer
    writer.close()

    # Save the trained model
    model_path = os.path.join(folder_path, "model.pth")
    torch.save(model.state_dict(), model_path)

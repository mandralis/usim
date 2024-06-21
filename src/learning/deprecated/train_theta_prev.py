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

# # Create a list to store previous Thetas
# Theta_prev_list = []

# # Add noise and shift Theta to create the previous Theta tensors
# n_prev_thetas = 1
# for i in range(1, n_prev_thetas+1):  # 1 to 10 previous thetas
#     Theta_prev = torch.zeros_like(Theta)
#     Theta_prev[i:] = Theta[:-i]  # Shift Theta down by i rows
#     # noise = torch.randn_like(Theta_prev) * 0.2  # Generate random noise
#     # Theta_prev += noise  # Add noise to Theta_prev
#     Theta_prev_list.append(Theta_prev)

# # Concatenate all previous Thetas along the second dimension
# Theta_prev_concat = torch.cat(Theta_prev_list, dim=1)

# # Concatenate X with all the previous Thetas
# X = torch.cat((X, Theta_prev_concat), dim=1)

# Verify the shapes
print(f"X shape: {X.shape}")
print(f"Theta shape: {Theta.shape}")

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
    def __init__(self, dim_in, dim_out, dim_prev_theta, layer_dims=[512,256,128,64], kernel_size=3, stride=1, padding=1):
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
        self.fc = nn.Linear(layer_dims[3] + dim_prev_theta , dim_out)
        self.dim_prev_theta = dim_prev_theta

    def forward(self, x_):
        # get waveform and previous theta
        # x = x_[:,:-self.dim_prev_theta]
        # theta_prev = x_[:,-self.dim_prev_theta:]
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
        # x = torch.cat((x, theta_prev), dim=1)
        x = self.fc(x)
        return x

def custom_loss(pred_theta, true_theta, prev_theta):
    mse_loss = nn.MSELoss()
    loss1 = mse_loss(pred_theta, true_theta)
    # loss2 = mse_loss(pred_theta, prev_theta)
    # return loss1 + 0*loss2
    return loss1

if __name__ == '__main__':
    # Set hyperparameters
    dim_n_joints = 8
    dim_prev_theta = dim_n_joints * 0
    batch_size = 128
    layer_dims = [256,128,64,32]
    kernel_size = 3
    dim_in = X.shape[1] - dim_prev_theta
    dim_out = Theta.shape[1]
    learning_rate = 0.001
    num_epochs = 1500
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

    # Create an instance of the ConvNet model
    model = ConvNet(dim_in, dim_out, dim_prev_theta, layer_dims, kernel_size=kernel_size)

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
                inputs = inputs.float()
                targets = targets.float()
                outputs = model(inputs)
                loss = criterion(outputs, targets, inputs[:,-dim_prev_theta:])

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
                    val_loss += criterion(outputs, targets, inputs[:,-dim_prev_theta:]).item()

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

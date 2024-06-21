import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from train_convlstm import ConvLSTMNet, TrainingDataset, DataLoader, custom_loss

# Train folder path
train_path = "/Users/imandralis/src/usim/src/learning/learned_models/20240620-170109"

# Load data
train_data           = torch.load(os.path.join(train_path, 'train_data.pth'))
val_data             = torch.load(os.path.join(train_path, 'val_data.pth'))
X, Theta             = train_data['X'].cpu(), train_data['Theta'].cpu()
X_train, Theta_train = train_data['X_train'].cpu(), train_data['Theta_train'].cpu()

# Load config
config               = torch.load(os.path.join(train_path, 'config.pth'))

# Load the model
model = ConvLSTMNet(config.get('dim_in'),config.get('dim_out'),config.get('lstm_hidden_size'),config.get('lstm_num_layers'),config.get('layer_dims'))
model.load_state_dict(torch.load(os.path.join(train_path, 'model.pth')))
model.eval()

# Create training dataset and dataloader
train_dataset = TrainingDataset(X_train, Theta_train, config.get('seq_length'))
train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size'), shuffle=False)

batch_size = config.get('batch_size')
print(batch_size)

# Predict on the training set
criterion = custom_loss
Theta_predicted = torch.zeros(len(train_loader), config.get('dim_out'))
with torch.no_grad():
    i = 0
    train_loss = 0.0
    for inputs, targets in train_loader:
        # Forward pass
        inputs = inputs.float()
        targets = targets.float()
        outputs = model(inputs)
        try:
            Theta_predicted[i*batch_size:(i+1)*batch_size,:] = outputs
        except:
            break
        embed()
        loss = criterion(outputs, targets[:,-1,:])
        train_loss += loss.item()
        i += 1
    train_loss /= len(train_loader)
    print(f'Training loss: {train_loss}')


# Plot the results on the training set
plt.figure(figsize=(12, 6))
for i in range(config.get('dim_out')):
    plt.subplot(3, 3, i+1)
    plt.plot(Theta_train[config.get('seq_length')-1:, i].detach().numpy(), 'b', label='Actual')
    plt.plot(Theta_predicted[:, i].detach().numpy(), 'r', label='Predicted')
    plt.xlabel('Sample')
    plt.ylabel('Angle')
    plt.title(f'Angle {i+1}')
    plt.legend()
plt.tight_layout()

plt.show()
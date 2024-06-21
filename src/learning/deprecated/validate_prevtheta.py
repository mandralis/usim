import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from IPython import embed
from train import ConvNet, TrainingDataset

# Load data
train_data           = torch.load('train_data.pth')
val_data             = torch.load('val_data.pth')
X_train, Theta_train = train_data['X'], train_data['Theta']
X_val, Theta_val     = val_data['X'], val_data['Theta']

# dimensions
dim_n_joints  = 8
dim_prev_thetas = dim_n_joints * 0
dim_in        = X.shape[1] - dim_prev_thetas
dim_out       = Theta.shape[1]

# Load the model
model_path = "/Users/imandralis/src/usim/src/learning/learned_models/20240620-161907"
model = ConvNet(dim_in,dim_out,dim_prev_thetas,[256,128,64,32])
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

# Unnormalize Theta and Theta_predicted
Theta_unnorm = Theta[:batch_size] * Theta_std + Theta_mean
Theta_predicted_unnorm = Theta_predicted * Theta_std + Theta_mean

# Compute covariance matrix
Theta_unnorm_np = Theta_unnorm.detach().numpy()
Theta_predicted_unnorm_np = Theta_predicted_unnorm.detach().numpy()
cov_matrix = np.cov(Theta_predicted_unnorm_np.T, Theta_unnorm_np.T)

# Extract the relevant part of the covariance matrix
dim = Theta_unnorm_np.shape[1]
cov_matrix = cov_matrix[:dim, dim:]

print("Covariance matrix between Theta_predicted and Theta:")
print(cov_matrix)

# Plot the results
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

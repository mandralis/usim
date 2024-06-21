# add imports
import torch.nn as nn


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
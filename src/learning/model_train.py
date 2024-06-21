
import torch

import copy
from torch import nn,optim 
from torchinfo import summary 
from models.model.transformer import Transformer 
from usim_dataset import myDataLoader
import numpy as np 
 
def cross_entropy_loss(pred, target):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(pred, target ) 
    return loss

def calc_loss_and_score(pred, target, metrics): 
    softmax = nn.Softmax(dim=1)

    pred =  pred.squeeze( -1)
    target= target.squeeze( -1)
    
    ce_loss = cross_entropy_loss(pred, target)
    metrics['loss'] .append( ce_loss.item() )
    pred = softmax(pred )
    
    _,pred = torch.max(pred, dim=1)
    metrics['correct']  += torch.sum(pred ==target ).item()
    metrics['total']  += target.size(0) 

    return ce_loss
 
def print_metrics(main_metrics_train,main_metrics_val,metrics, phase):
   
    correct= metrics['correct']  
    total= metrics['total']  
    accuracy = 100*correct / total
    loss= metrics['loss'] 
    if(phase == 'train'):
        main_metrics_train['loss'].append( np.mean(loss)) 
        main_metrics_train['accuracy'].append( accuracy ) 
    else:
        main_metrics_val['loss'].append(np.mean(loss)) 
        main_metrics_val['accuracy'].append(accuracy ) 
    
    result = "phase: "+str(phase) \
    +  ' \nloss : {:4f}'.format(np.mean(loss))   +    ' accuracy : {:4f}'.format(accuracy)        +"\n"
    return result 

def train_model(dataloaders,model,optimizer, num_epochs=100): 
 
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    train_dict= dict()
    train_dict['loss']= list()
    train_dict['accuracy']= list() 
    val_dict= dict()
    val_dict['loss']= list()
    val_dict['accuracy']= list() 

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10) 

        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = dict()
            metrics['loss']=list()
            metrics['correct']=0
            metrics['total']=0
 
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device=device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.int)
                # zero the parameter gradients
                optimizer.zero_grad()


                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    #print('outputs size: '+ str(outputs.size()) )
                    loss = calc_loss_and_score(outputs, labels, metrics)   
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                #print('epoch samples: '+ str(epoch_samples)) 
            print(print_metrics(main_metrics_train=train_dict, main_metrics_val=val_dict,metrics=metrics,phase=phase ))
            epoch_loss = np.mean(metrics['loss'])
        
            if phase == 'val' and epoch_loss < best_loss:
                    print("saving best model")
                    best_loss = epoch_loss 

    print('Best val loss: {:4f}'.format(best_loss))

from sklearn.model_selection import train_test_split
import numpy as np 

import torch.nn as nn
import torch.optim as optim
import scipy.io as sio

# load data
X     = sio.loadmat('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/X.mat')['X'][:,:2000]
Theta = sio.loadmat('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/Theta_relative_8_joints.mat')['Theta_relative']

# Convert data to tensor and float32
X = torch.tensor(X).float()
Theta = torch.tensor(Theta).float()

# Normalize the input data and output data
X_mean = torch.mean(X, dim=0)
X_std = torch.std(X, dim=0) + 1e-6
X = (X - X_mean) / X_std

Theta_mean = torch.mean(Theta, dim=0)
Theta_std = torch.std(Theta, dim=0) + 1e-6
Theta = (Theta - Theta_mean) / Theta_std

validation_split = 0.2  # 20% of the data will be used for validation

# Create train-validation split
X_train, X_val, Theta_train, Theta_val = train_test_split(X, Theta, test_size=validation_split)

batch_size = 32
dataloaders= myDataLoader(batch_size,X_train,Theta_train,X_val,Theta_val).getDataLoader()

device = torch.device("cpu")
sequence_len=X.shape[1] # sequence length of time series
max_len=X.shape[1] # max time series sequence length 
n_head = 2 # number of attention head
n_layer = 1# number of encoder layer
drop_prob = 0.1
d_model = X.shape[1] # number of dimension ( for positional embedding)
ffn_hidden = 128 # size of hidden layer before classification 
feature = 1 # for univariate time series (1d), it must be adjusted for 1. 
model =  Transformer(  d_model=d_model, n_head=n_head, max_len=max_len, seq_len=sequence_len, ffn_hidden=ffn_hidden, n_layers=n_layer, drop_prob=drop_prob, details=False,device=device).to(device=device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

model_normal_ce = train_model(dataloaders=dataloaders,model=model,optimizer=optimizer, num_epochs=20)
torch.save(model.state_dict(), 'myModel')
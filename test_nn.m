clear all
close all
clc

%% import neural net
% Import the ONNX model as a dlnetwork
net = importONNXNetwork('model.onnx', 'InputDataFormats', {'BC'}, 'TargetNetwork', 'dlnetwork');

%% load
load('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/X.mat');
load('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/Px_array.mat');
load('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/Py_array.mat');
load('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/Theta_relative.mat');

%% get normalized input data

% Normalize the input data X
X_mean = mean(X, 1);  % Compute mean along the first dimension (rows)
X_std = std(X, 1);    % Compute standard deviation along the first dimension (rows)
X_normalized = (X - X_mean) ./ (X_std + 1e-6);

% Normalize the output data Theta
Theta_relative_mean = mean(Theta_relative, 1);  % Compute mean along the first dimension (rows)
Theta_relative_std = std(Theta_relative, 1);    % Compute standard deviation along the first dimension (rows)
Theta_relative_normalized = (Theta_relative - Theta_relative_mean) ./ (Theta_relative_std + 1e-6);


%% get wire length in pixels
L = Px(1,end);

%% get wire clamped position in y
py_clamp = Py(1,1);

%% number of joints on kinematic linkage
N_joints = 4;

%% split wire in 4 equal segments
a = L/N_joints;

%% test neural network against data 

Theta_predicted = zeros(size(Theta_relative));
for i = 1:size(Px,1)

    % Convert input data to a dlarray
    dlInput = dlarray(X_normalized(i,1:2000), 'BC');
    Theta_relative_normalized = extractdata(predict(net, dlInput));

    % unnormalize the output
    Theta_relative_ = Theta_relative_normalized .* (Theta_relative_std' + 1e-6) + Theta_relative_mean';

    Theta_predicted(i,:) = Theta_relative_;

    % plot kinematic linkage with given angles
    plot(Px(i,:),Py(i,:),"Color",'b',LineWidth=1.0);
    axis([0,L,0,1080])
    hold on
    Pkin = forward_kin(0,a,a,a,a,Theta_relative_(1),Theta_relative_(2),Theta_relative_(3),Theta_relative_(4));
    plot(Pkin(2,:),Pkin(1,:) + py_clamp,'Marker','o','MarkerFaceColor','k',"Color",'r','MarkerEdgeColor','k',LineWidth=1.0);
    pause(0.01);
    clf;
end


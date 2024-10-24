clear all
close all
clc

%% import neural net
% Import the ONNX model as a dlnetwork
net = importONNXNetwork('C:\Users\arosa\Box\USS Catheter\data\data_09_27_2024_15_40_55\model1.onnx', 'InputDataFormats', {'BC'}, 'TargetNetwork', 'dlnetwork');

%% load
load('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_09_27_2024_15_40_55/X.mat');
load('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_09_27_2024_15_40_55/Px_array.mat');
load('/Users/imandralis/Library/CloudStorage/Box-Bo  x/USS Catheter/data/data_09_27_2024_15_40_55/Py_array.mat');
load('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_09_27_2024_15_40_55/Theta_relative_8_joints.mat');

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
N_joints = 8;

%% split wire in N_joints equal segments
a_ = L/N_joints;
a = [0, a_ * ones(1,N_joints)];

%% test neural network against data 

% init filter
alpha = 0.7;
filter = ExponentialSmoothingFilter(N_joints,alpha);
% N_window = 1;
% filter = MovingAverageFilter(N_joints,N_window);
% q = 1;
% r = 1e-2;
% filter = KalmanFilter(N_joints,q,r);

Theta_predicted = zeros(size(Theta_relative));
for i = 1:size(Px,1)
    
    % get current starting position
    py_clamp = Py(i,1);

    % Convert input data to a dlarray
    dlInput = dlarray(X_normalized(i,1:2000), 'BC');
    Theta_relative_normalized = extractdata(predict(net, dlInput));

    % unnormalize the output
    Theta_relative_ = Theta_relative_normalized .* (Theta_relative_std' + 1e-6) + Theta_relative_mean';

    % filter with kalman filter
    Theta_relative_filtered = filter.update(Theta_relative_);

    Theta_predicted(i,:) = Theta_relative_filtered;

    % plot kinematic linkage with given angles
    plot(Px(i,:),Py(i,:),"Color",'b',LineWidth=1.0);
    axis([0,L,0,1080])
    hold on
    Pkin = forward_kin(a,Theta_relative_filtered');
    Pkin_unfiltered = forward_kin(a,Theta_relative_');

    plot(Pkin(2,:),Pkin(1,:) + py_clamp,'Marker','o','MarkerFaceColor','k',"Color",'r','MarkerEdgeColor','k',LineWidth=1.0);
    plot(Pkin_unfiltered(2,:),Pkin_unfiltered(1,:) + py_clamp,'Marker','o','MarkerFaceColor','k',"Color",'g','MarkerEdgeColor','k',LineWidth=1.0);

    pause(0.01);
    clf;
end


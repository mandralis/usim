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
load('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/Theta_relative_8_joints.mat');
load('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/acquisition_params.mat')

%% get time array
t_total = (acquisition_params.t_per_acquisition - 0.5) * acquisition_params.n_acquisition_cycles_max;
t = linspace(0,t_total,size(X,1));

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

% initialize filter
q = 0.5;
r = 0.5;
filter = KalmanFilter(N_joints,q,r);

Theta_predicted = zeros(size(Theta_relative));

n_joints = 8;
% n_prev_theta = 1;
% Theta_relative_normalized_prev = zeros(n_prev_theta,n_joints);
Theta_relative_mean = zeros(1,n_joints);
for i = 1:size(Px,1)
    % get current starting position
    py_clamp = Py(i,1);

    % Convert input data to a dlarray
%     dlInput = dlarray([X_normalized(i,1:2000),reshape(Theta_relative_normalized_prev,1,n_prev_theta*n_joints)], 'BC');
    dlInput = dlarray(X_normalized(i,1:2000), 'BC');
    Theta_relative_normalized = extractdata(predict(net, dlInput));

%     Theta_relative_normalized_prev(1,:) = [];
%     Theta_relative_normalized_prev = [Theta_relative_normalized_prev;Theta_relative_normalized'];
%     Theta_relative_normalized_prev = Theta_relative_normalized';

    % unnormalize the output
    Theta_relative_ = Theta_relative_normalized .* (Theta_relative_std' + 1e-6) + Theta_relative_mean';

    Theta_relative_mean = Theta_relative_mean + Theta_relative_';

    % filter with kalman filter
    Theta_relative_filtered = filter.update(Theta_relative_);

%     Theta_predicted(i,:) = Theta_relative_filtered;

    % plot kinematic linkage with given angles
    plot(Px(i,:),Py(i,:),"Color",'b',LineWidth=1.0);
    axis([0,L,0,1080])
    hold on
    if mod(i,1) == 0
%         Theta_relative_mean = Theta_relative_mean / 20;

          Pkin = forward_kin(a,Theta_relative_filtered');
%         Pkin_unfiltered = forward_kin(a,Theta_relative_mean);
    
        plot(Pkin(2,:),Pkin(1,:) + py_clamp,'Marker','o','MarkerFaceColor','k',"Color",'r','MarkerEdgeColor','k',LineWidth=1.0);
%         plot(Pkin_unfiltered(2,:),Pkin_unfiltered(1,:) + py_clamp,'Marker','o','MarkerFaceColor','k',"Color",'r','MarkerEdgeColor','k',LineWidth=1.0);
    
        Theta_relative_mean = zeros(1,8);
    end
    pause(0.005);
    clf;
end


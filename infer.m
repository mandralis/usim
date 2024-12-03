clear all
close all
clc

%% import neural net
% Import the ONNX model as a dlnetwork
net = importONNXNetwork('C:\Users\arosa\Box\USS Catheter\data\data_10_24_2024_16_38_25\model2.onnx', 'InputDataFormats', {'BC'}, 'TargetNetwork', 'dlnetwork');

%% load
load('C:\Users\arosa\Box\USS Catheter\data\data_10_24_2024_16_38_25/X.mat');
load('C:\Users\arosa\Box\USS Catheter\data\data_10_24_2024_16_38_25/Px_array.mat');
load('C:\Users\arosa\Box\USS Catheter\data\data_10_24_2024_16_38_25/Py_array.mat');
load('C:\Users\arosa\Box\USS Catheter\data\data_10_24_2024_16_38_25/Theta_relative.mat');
load('C:\Users\arosa\Box\USS Catheter\data\data_10_24_2024_16_38_25/acquisition_params.mat')

%% get correct bounds for input X
nx_start = 200 + 1;
nx_end   = 1200;

%% get time array
t_total = (acquisition_params.t_per_acquisition - 0.5) * acquisition_params.n_acquisition_cycles_max;
t = linspace(0,t_total,size(X,1));
dt = t(2)-t(1);

%% get normalized input data

% remove nans from theta and also remove from same indices in X
Theta_relative_cleaned = Theta_relative(~any(isnan(Theta_relative), 2), :);
X_cleaned              = X(~any(isnan(Theta_relative), 2), :);

% assign back to values
Theta_relative = Theta_relative_cleaned;
X = X_cleaned;

% Normalize the input data X
X_mean = mean(X, 1);  % Compute mean along the first dimension (rows)
X_std = std(X, 1);    % Compute standard deviation along the first dimension (rows)
X_normalized = (X - X_mean) ./ (X_std + 1e-6);

% Normalize the output data Theta
Theta_relative_mean = mean(Theta_relative, 1);  % Compute mean along the first dimension (rows)
Theta_relative_std = std(Theta_relative, 1);    % Compute standard deviation along the first dimension (rows)
Theta_relative_normalized = (Theta_relative - Theta_relative_mean) ./ (Theta_relative_std + 1e-6);


%% get wire length in pixels
L = max(Px(1,:));

%% get wire clamped position in y
py_clamp = Py(1,1);

%% number of joints on kinematic linkage
n_joints = 9;

%% split wire in N_joints equal segments
a_ = L/n_joints;
a = [0, a_ * ones(1,n_joints)];

%% infer on neural network
% connect matlab to oscilloscope 
myScope = connect();
Theta_predicted = zeros(size(Theta_relative));
figure();
hold on;
while(1)    
    X_unnormalized = getWaveform(myScope, 'acquisition', true);
    clf;
    X_ = (X_unnormalized - X_mean(nx_start:nx_end)) ./ (X_std(nx_start:nx_end) + 1e-6);

    % Convert input data to a dlarray
    dlInput = dlarray(X_, 'BC');
    Theta_relative_normalized = extractdata(predict(net, dlInput));

    % unnormalize the output
    Theta_relative_ = Theta_relative_normalized .* (Theta_relative_std' + 1e-6) + Theta_relative_mean';

    % get and plot the predicted shape
    Pkin = forward_kin(a,2*Theta_relative_');
%     subplot(2,1,1)
    plot(Pkin(2,:),Pkin(1,:) + py_clamp,'Marker','o','MarkerFaceColor','k',"Color",'r','MarkerEdgeColor','k',LineWidth=1.0);
    axis([0,L,0,1080]);
%     subplot(2,1,2)
%     plot(X_)
    pause(0.01)
    
end


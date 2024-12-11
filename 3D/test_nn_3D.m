clear all
close all
clc

%% import neural net
% Import the ONNX model as a dlnetwork
net = importONNXNetwork('C:\Users\arosa\Box\USS Catheter\data\data_09_27_2024_15_40_55\model1.onnx', 'InputDataFormats', {'BC'}, 'TargetNetwork', 'dlnetwork');

%% load
load('X.mat');
load('position_mat_tot.amt');
load('Theta_relative.mat');
load('acquisition_params.mat')

%% get correct bounds for input X
nx_start = 200 + 1;
nx_end   = 1000;

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


%% get wire length in pixels (MODIFY)
L = max(Px(1,:));

%% get wire clamped position in y
py_clamp = Py(1,1);

%% number of joints on kinematic linkage
n_joints = 10;

%% split wire in N_joints equal segments (MODIFY)
a_ = L/n_joints;
a = [0, a_ * ones(1,n_joints)];

%% test neural network against data 

% visualization parameters
update_period = dt;
update_freq   = ceil(dt/dt);

% initialize filter
filter = IdentityFilter();

% visualization
filtering       = true;
Theta_predicted = zeros(size(Theta_relative));
figure();
hold on;
for i = 1:size(Px,1)
    % get current starting position (MODIFY)
    py_clamp = Py(i,1);

    % Convert input data to a dlarray
    dlInput = dlarray(X_normalized(i,nx_start:nx_end), 'BC');
    Theta_relative_normalized = extractdata(predict(net, dlInput));

    % unnormalize the output
    Theta_relative_ = Theta_relative_normalized .* (Theta_relative_std' + 1e-6) + Theta_relative_mean';

    % filter
    if filtering == true
        Theta_relative_filtered = filter.update(Theta_relative_);
    else
        Theta_relative_filtered = Theta_relative_;
    end          

    % plot kinematic linkage with given angles (MODIFY)
    max_x = max(Px(i,:));
    plot(Px(i,1:max_x),Py(i,1:max_x),"Color",'b',LineWidth=1.0);
    axis([0,L,0,1080])
    hold on

    % at update rate plot the model prediction
    if mod(i,1) == 0
        % get and plot the predicted shape
        Pkin = forward_kin(a,Theta_relative_filtered');
        plot(Pkin(2,:),Pkin(1,:) + py_clamp,'Marker','o','MarkerFaceColor','k',"Color",'r','MarkerEdgeColor','k',LineWidth=1.0);
    end

    % pause for time dt to get a (quasi) real time loop 
    pause(dt);
    clf;
end


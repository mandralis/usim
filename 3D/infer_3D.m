clear all
close all
clc

%% import neural net
% Import the ONNX model as a dlnetwork
net = importONNXNetwork('C:\Users\arosa\Box\USS Catheter\3d_data\data_11_08_2024_16_47_09\model.onnx', 'InputDataFormats', {'BC'}, 'TargetNetwork', 'dlnetwork');

%% setup osc
myScope = connect();

%% load
load('C:\Users\arosa\Box\USS Catheter\3d_data\data_11_08_2024_16_47_09\X.mat');
% load('C:\Users\arosa\Box\USS Catheter\3d_data\data_11_08_2024_16_47_09\position_matrix_tot.mat');
load('C:\Users\arosa\Box\USS Catheter\3d_data\data_11_08_2024_16_47_09\Theta_relative_3D.mat');
load('C:\Users\arosa\Box\USS Catheter\3d_data\data_11_08_2024_16_47_09\acquisition_params.mat')

%% get correct bounds for input X
nx_start = 200 + 1;
nx_end   = 1000;

%% get time array
t_total = (acquisition_params.t_per_acquisition - 0.5) * acquisition_params.n_acquisition_cycles_max;
t = linspace(0,t_total,size(X,1));
dt = t(2)-t(1);

%% get normalized input data

% remove nans from theta and also remove from same indices in X
Theta_relative_cleaned = Theta_relative_3D(~any(isnan(Theta_relative_3D), 2), :);
X_cleaned              = X(~any(isnan(Theta_relative_3D), 2), :);

% assign back to values
Theta_relative_3D = Theta_relative_cleaned;
X = X_cleaned;

% Normalize the input data X
X_mean = mean(X, 1);  % Compute mean along the first dimension (rows)
X_std = std(X, 1);    % Compute standard deviation along the first dimension (rows)
X_normalized = (X - X_mean) ./ (X_std + 1e-6);

% Normalize the output data Theta
Theta_relative_mean = mean(Theta_relative_3D, 1);  % Compute mean along the first dimension (rows)
Theta_relative_std = std(Theta_relative_3D, 1);    % Compute standard deviation along the first dimension (rows)
Theta_relative_normalized = (Theta_relative_3D - Theta_relative_mean) ./ (Theta_relative_std + 1e-6);


%% get wire length in pixels (MODIFY)
L = 0.5; %this is arbitrary 

%% get wire clamped position in y
% py_clamp = Py(1,1);

%% number of joints on kinematic linkage
n_joints = 5; %there are actually 2x this many but each is a double joint (one around xy, one around xz)

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
point_num = 1; %which point is the zero position
filtering       = true;
Theta_predicted = zeros(size(Theta_relative_3D));
figure();
hold on;
i = 0;
while (1)
    % get current starting position (MODIFY)
    %     py_clamp = Py(i,1);
    i = i+1;
    % get waveform from osc
    X_unnormalized = getWaveform(myScope, 'acquisition', true);
    X_unnormalized = X_unnormalized(nx_start:nx_end);
    X_ = (X_unnormalized - X_mean(nx_start:nx_end)) ./ (X_std(nx_start:nx_end) + 1e-6);


    % Convert input data to a dlarray
    dlInput = dlarray(X_, 'BC');
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
%     max_x = max(Px(i,:));
%     plot(Px(i,1:max_x),Py(i,1:max_x),"Color",'b',LineWidth=1.0);
%     axis([0,L,0,1080])
%     hold on

    % at update rate plot the model prediction
    if mod(i,1) == 0
        
        % get and plot the predicted shape
        subplot(2,1,1)
%         scatter(Position_matrix_tot(1,:,i)-Position_matrix_tot(1,point_num,i),Position_matrix_tot(2,:,i)-Position_matrix_tot(2,point_num,i))
        axis([-0.3,0.7,-.3,0.3])
        hold on
%         Pkin = forward_kin5(a,Theta_relative_3D(i,1:5));
%         plot(Pkin(2,:)-Pkin(2,1),Pkin(1,:)-Pkin(1,1),'Marker','o','MarkerFaceColor','k',"Color",'r','MarkerEdgeColor','k',LineWidth=1.0);
        Pkin = forward_kin5(a,Theta_relative_filtered(1:5)');
        plot(Pkin(2,:)-Pkin(2,1),Pkin(1,:)-Pkin(1,1),'Marker','o','MarkerFaceColor','k',"Color",'b','MarkerEdgeColor','k',LineWidth=1.0);        
        
        
        subplot(2,1,2)
%         scatter(Position_matrix_tot(1,:,i)-Position_matrix_tot(1,point_num,i),Position_matrix_tot(3,:,i)-Position_matrix_tot(3,point_num,i))
        axis([-0.3,0.7,-.3,0.3])
        hold on
%         Pkin = forward_kin5(a,Theta_relative_3D(i,6:10));
%         plot(Pkin(2,:)-Pkin(2,1),Pkin(1,:)-Pkin(1,1),'Marker','o','MarkerFaceColor','k',"Color",'r','MarkerEdgeColor','k',LineWidth=1.0);
        Pkin = forward_kin5(a,Theta_relative_filtered(6:10)');
        plot(Pkin(2,:)-Pkin(2,1),Pkin(1,:)-Pkin(1,1),'Marker','o','MarkerFaceColor','k',"Color",'b','MarkerEdgeColor','k',LineWidth=1.0);


    
    end

    
    drawnow
    pause(0.01)
    clf
%     % pause for time dt to get a (quasi) real time loop
%     
%     pause(dt);
%     clf;
end


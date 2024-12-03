clear all
close all
clc

%% load
load('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/X.mat');
load('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/Px_array.mat');
load('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/Py_array.mat');
load('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/Theta_relative_4_joints.mat');
load('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/ampl.mat')

%% get wire length in pixels
L = Px(1,end);

%% get wire clamped position in y
py_clamp = Py(1,1);

%% number of joints on kinematic linkage
N_joints = 4;

%% split wire in N_joints equal segments
a_ = L/N_joints;
a = [0, a_ * ones(1,N_joints)];

%% test linear regression against data 

% augment amplitude to fit a bias
ampl_aug = zeros(size(ampl,1),size(ampl,2)+1);
for i=1:size(ampl,1)
    ampl_aug(i,:) = [1,ampl(i,:)];
end

% get parameters
fit = pinv(ampl_aug) * Theta_relative;

Theta_relative_predicted = zeros(size(Theta_relative));
for i = 1:size(Px,1)
    % predict theta
    Theta_relative_predicted_ = ampl_aug(i,:) * fit;

    Theta_relative_predicted(i,:) = Theta_relative_predicted_;

    % plot kinematic linkage with given angles
    plot(Px(i,:),Py(i,:),"Color",'b',LineWidth=1.0);
    axis([0,L,0,1080])
    hold on
    Pkin = forward_kin(a,Theta_relative_predicted_);
    plot(Pkin(2,:),Pkin(1,:) + py_clamp,'Marker','o','MarkerFaceColor','k',"Color",'r','MarkerEdgeColor','k',LineWidth=1.0);
    pause(0.01);
    clf;
end


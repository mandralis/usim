clear all
close all
clc

%% load
load('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_09_27_2024_15_40_55/Px_array.mat');
load('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_09_27_2024_15_40_55/Py_array.mat');
% load('C:\Users\arosa\Desktop\matlab.mat')

%% get number of data points
N_samples = size(Px,1);

%% get wire length in pixels
L = max(Px(1,:)); % we assume wire is straight in the first frame

%% get wire clamped position in y
py_clamp = Py(1,1);

%% number of joints on kinematic linkage
N_joints = 8;

%% split wire in n_joints equal segments
a_ = L/N_joints;
a = [0, a_ * ones(1,N_joints)];

%% split wire into segments to get angles
% query_x = floor(linspace(1,L,N_joints+1));
Theta = zeros(N_samples,N_joints);
Theta_relative = zeros(N_samples,N_joints);
figure();
for i = 1:size(Px,1)
    disp(i)
    % get current starting y position
    py_clamp = Py(i,1);

    % get query points by integrating current arc length
    dx = diff(Px(i,:));
    dy = diff(Py(i,:));
    arclength = cumtrapz(sqrt(1 + dy.^2)); % dx spacing is all one

    % find where arclength becomes greater than multiples of "a"
    query_x = [];
    for ii = 1:N_joints
        idx = find(arclength>a_*ii,1);
        if isempty(idx)
            query_x = [query_x, L];
        else
            query_x = [query_x idx];
        end
    end
    query_x = [1,query_x];

    % get y positions at query point
    query_y = Py(i,query_x);

    % get angle 
    Theta(i,:) = -atan(diff(query_y)./diff(query_x));

    % convert to relative angle for kinematic linkage
    Theta_relative(i,:) = [Theta(i,1), diff(Theta(i,:))];

%     % plot kinematic linkage with given angles
%     plot(Px(i,:),Py(i,:),"Color",'b',LineWidth=1.0);
%     axis([0,L,0,1080])
%     hold on
%     Pkin = forward_kin(a,Theta_relative(i,:));
%     plot(Pkin(2,:),Pkin(1,:) + py_clamp,'Marker','o','MarkerFaceColor','k',"Color",'r','MarkerEdgeColor','k',LineWidth=1.0);
%     pause(0.01);
%     clf;
end

save('Theta_relative.mat','Theta_relative')


clear all
close all
clc

%% load
load('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/X.mat');
load('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/Y.mat');

%% remove unecessary data
X_cut = X(:,1:2000);
plot(X_cut(100,:))

%% plot
for i = 1:size(X_cut,1)
    plot(X(i,:))
    pause(0.01)
end

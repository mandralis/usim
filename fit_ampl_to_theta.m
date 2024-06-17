clear all
close all
clc

% linear regression to map amplitudes to angles 
load('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/ampl.mat')
load('/Users/imandralis/Library/CloudStorage/Box-Box/USS Catheter/data/data_05_26_2024_16_54_50/Theta_no_arclength.mat')

% augment amplitude to fit a bias
ampl_aug = zeros(size(ampl,1),size(ampl,2)+1);
for i=1:size(ampl,1)
    ampl_aug(i,:) = [1,ampl(i,:)];
end

% get parameters
fit = pinv(ampl_aug) * Theta;
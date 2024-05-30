% clear workspace
clear all
close all
clc

data_path = 'data_05_26_2024_16_54_50/';
load([data_path,'params.mat']);

% get key parameters
n_acquisition_cycles             = params.n_acquisition_cycles;
n_samples_per_acquisition_cycle  = params.n_samples_per_acquisition_cycle;
n_triggers_per_acquisition_cycle = params.n_triggers_per_acquisition_cycle;

% get derived parameters
n_waveforms            = n_acquisition_cycles*n_triggers_per_acquisition_cycle;
n_samples_per_waveform = n_samples_per_acquisition_cycle/n_triggers_per_acquisition_cycle;

% define the data frame 
X = zeros(n_waveforms,n_samples_per_waveform);
for i = 1:n_acquisition_cycles
    % display counter
    disp(i)

    % load one acquisition cycle worth of data
    load([data_path,'pulse_',num2str(i),'.mat']);
    
    % get start and end indices
    start_idx = 1 + (i-1)*n_triggers_per_acquisition_cycle;
    end_idx   = i*n_triggers_per_acquisition_cycle;
    
    % put reshaped data into data frame
    X(start_idx:end_idx,:) = reshape(w,n_samples_per_waveform,n_triggers_per_acquisition_cycle)';
end

% save data 
save([data_path,'X.mat'],'X');
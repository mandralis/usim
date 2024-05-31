% TEST

% clear workspace
clear all
close all
clc

data_path = 'data_05_26_2024_16_54_50/';
load([data_path,'acquisition_params.mat']);

addpath([get_local_data_path(),data_path])

% get acquisition parameters
n_acquisition_cycles             = acquisition_params.n_acquisition_cycles;
n_samples_per_acquisition_cycle  = acquisition_params.n_samples_per_acquisition_cycle;
n_triggers_per_acquisition_cycle = acquisition_params.n_triggers_per_acquisition_cycle;

% instantiate PostProcessingParameters object
post_processing_params = PostProcessingParameters();

% get key parameters
mask_threshold      = post_processing_params.mask_threshold;
image_crop_array    = post_processing_params.image_crop_array;
smooth_before_fit   = post_processing_params.smooth_before_fit;
spike_detect_thresh = post_processing_params.spike_detect_thresh;
noise_floor         = post_processing_params.noise_floor;
zero_length         = post_processing_params.zero_length;


% get derived parameters
n_valid_triggers_per_acquisition_cycle = n_triggers_per_acquisition_cycle - 1; 
n_waveforms                            = n_acquisition_cycles * n_valid_triggers_per_acquisition_cycle;
n_samples_per_waveform                 = n_samples_per_acquisition_cycle/n_triggers_per_acquisition_cycle;

% define the data frame 
X = zeros(n_waveforms,n_samples_per_waveform);
for i = 1:n_acquisition_cycles
    % display counter
    disp(i)

    % load one acquisition cycle worth of data
    load([data_path,'pulse_',num2str(i),'.mat']);
    
    % get start and end indices
    start_idx = 1 + (i-1)*n_valid_triggers_per_acquisition_cycle;
    end_idx   = i*n_valid_triggers_per_acquisition_cycle;
    
    % put reshaped data into data frame
    start_pos = get_start_of_pulse(w,n_samples_per_waveform,spike_detect_thresh,noise_floor,zero_length);
  
    w_phase_locked = w(start_pos:end -(n_samples_per_waveform-start_pos)-1);
    X(start_idx:end_idx,:) = reshape(w_phase_locked,n_samples_per_waveform,n_valid_triggers_per_acquisition_cycle)';
end

% save waveform data 
% save([data_path,'X.mat'],'X');

% % get the curvature and position arrays
% [curvature_array,x_array,y_array] = getCurvatureAndPositionArrays(data_path,mask_threshold,image_crop_array,smooth_before_fit);
% 
% % curvature array and position arrays
% save([data_path,'Y.mat'],'curvature_array');
% save([data_path,'px_array.mat'],'x_array');
% save([data_path,'py_array.mat'],'y_array');

% save post_processing parameters
% save([fname,'/post_processing_params.mat'],'post_processing_params');

% plot if needed to verify the fit
% plotFitCurvatureImage(curvature_array,x_array,y_array)



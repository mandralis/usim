% TEST

% clear workspace
clear all
close all
clc

data_path = 'data_05_26_2024_16_54_50/';
load([data_path,'acquisition_params.mat']);

% get key parameters
n_acquisition_cycles             = acquisition_params.n_acquisition_cycles;
n_samples_per_acquisition_cycle  = acquisition_params.n_samples_per_acquisition_cycle;
n_triggers_per_acquisition_cycle = acquisition_params.n_triggers_per_acquisition_cycle;

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

% save waveform data 
save([data_path,'X.mat'],'X');

% instantiate PostProcessingParameters object
post_processing_params = PostProcessingParameters();

% get key parameters
mask_threshold    = post_processing_params.mask_threshold;
image_crop_array  = post_processing_params.image_crop_array;
smooth_before_fit = post_processing_params.smooth_before_fit;

% get the curvature and position arrays
[curvature_array,x_array,y_array] = getCurvatureAndPositionArrays(data_path,mask_threshold,image_crop_array,smooth_before_fit);

% curvature array and position arrays
save([data_path,'Y.mat'],'curvature_array');
save([data_path,'px_array.mat'],'x_array');
save([data_path,'py_array.mat'],'y_array');

% save post_processing parameters
save([fname,'/post_processing_params.mat'],'post_processing_params');

% plot if needed to verify the fit
% plotFitCurvatureImage(curvature_array,x_array,y_array)



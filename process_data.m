% clear workspace
clear all
close all
clc

% add necessary paths
addpath(genpath('src/'));
addpath(genpath('utils/'));

data_path = 'data_10_24_2024_16_38_25/';
im_data_folder = 'C:/Users/arosa/Desktop/Images_10_24_24\10_24_24';
addpath(genpath([get_local_data_path(),data_path]));
load([data_path,'acquisition_params.mat']);

% get acquisition parameters
n_acquisition_cycles             = acquisition_params.n_acquisition_cycles;
n_samples_per_acquisition_cycle  = acquisition_params.n_samples_per_acquisition_cycle;
n_triggers_per_acquisition_cycle = acquisition_params.n_triggers_per_acquisition_cycle;

% instantiate PostProcessingParameters object
post_processing_params = PostProcessingParameters();

% get key parameters
mask_threshold       = post_processing_params.mask_threshold;
image_crop_array     = post_processing_params.image_crop_array;
smooth_before_fit    = post_processing_params.smooth_before_fit;
spike_detect_thresh  = post_processing_params.spike_detect_thresh;
noise_floor          = post_processing_params.noise_floor;
zero_length          = post_processing_params.zero_length;
shadow_removal_array = post_processing_params.shadow_removal_array;
wire_length          = post_processing_params.wire_length;


% get derived parameters
n_valid_triggers_per_acquisition_cycle = n_triggers_per_acquisition_cycle - 1; 
n_waveforms                            = n_acquisition_cycles * n_valid_triggers_per_acquisition_cycle;
n_samples_per_waveform                 = n_samples_per_acquisition_cycle/n_triggers_per_acquisition_cycle;
n_length_of_curv_array                 = image_crop_array(4)-image_crop_array(3)+1;

% define the data frame 
X = zeros(n_waveforms,n_samples_per_waveform);
Y = zeros(n_waveforms,n_length_of_curv_array);
Px = zeros(n_waveforms,n_length_of_curv_array);
Py = zeros(n_waveforms,n_length_of_curv_array);

%%
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
    
    % get the curvature and position arrays
    %now do the curvature from the images for each acq cycle
    im_data_path = [im_data_folder,'/test_',num2str(i,'%03.f'),'/'];
    [curvature_array,x_array,y_array] = getCurvatureAndPositionArrays(im_data_path,mask_threshold,image_crop_array,smooth_before_fit,shadow_removal_array,wire_length);
    
    % put reshaped Curvature data into data frame
    Y(start_idx:end_idx,:) = curvature_array(1:end-1,:);
    Px(start_idx:end_idx,:) = x_array(1:end-1,:);
    Py(start_idx:end_idx,:) = y_array(1:end-1,:);
    %now remove the last frame and append to the end of a big list (this may be done already)
%     
    
    
end

%% fix X matrix to make everything line up
% probably a modified version of get start of pulse function

[a,b] = size(X);

for j = 1:a 
    
    ind = find(X(j,1:100) < spike_detect_thresh);
    position = ind(1);
    
    X(j,1:b-(position-1)) = X(j,position:end);
    
end
%%
% save waveform data Currently saves to the local directory cause im lazy 
%well it actually doesnt save at all lol
fname = [get_local_data_path(),data_path];
save([fname,'X.mat'],'X');

% curvature array and position arrays
save([fname,'Y.mat'],'Y');
save([fname,'Px_array.mat'],'Px');
save([fname,'Py_array.mat'],'Py');
% save('Y.mat','Y');
% save('Px_array.mat','Px');
% save('Py_array.mat','Py');

% save post_processing parameters
save([fname,'/post_processing_params.mat'],'post_processing_params');

%%

% plot if needed to verify the fit
plotFitCurvatureImage(curvature_array,x_array,y_array,image_crop_array)



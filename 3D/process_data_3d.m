% clear workspace
clear all
close all
clc
%%
% add necessary paths
addpath(genpath('src/'));
addpath(genpath('utils/'));

data_path = 'data_11_08_2024_16_47_09\';
fname = [get_local_data_path(),data_path];
addpath(genpath(fname));
load([fname,'acquisition_params.mat']);

% get acquisition parameters
n_acquisition_cycles             = acquisition_params.n_acquisition_cycles;
n_samples_per_acquisition_cycle  = acquisition_params.n_samples_per_acquisition_cycle;
n_triggers_per_acquisition_cycle = acquisition_params.n_triggers_per_acquisition_cycle;

% instantiate PostProcessingParameters object
post_processing_params = PostProcessingParameters();

% get key parameters
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
Theta_relative_zx = zeros(n_waveforms,5);
Theta_relative_yx = zeros(n_waveforms,5);

Position_matrix_tot = zeros(3,6,n_waveforms);


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

    % Get the angles for each aquisition cycle frame 
    Optitrack_matrix = import_3D_point_matrix([data_path,'ussc_',num2str(i,'%03.f'),'.csv']);
    Position_matrix = zeros(3,6,n_valid_triggers_per_acquisition_cycle);
    
    for j = 1:n_valid_triggers_per_acquisition_cycle
    
        Position_matrix(:,:,j) = reshape(Optitrack_matrix(j,3:20),3,6);
        
    end
    
    [TR_zx,TR_yx] = get_angles_from_positions_3d(Position_matrix,n_valid_triggers_per_acquisition_cycle);
    
    Theta_relative_zx((i-1)*n_valid_triggers_per_acquisition_cycle+1:((i)*n_valid_triggers_per_acquisition_cycle),:) = TR_zx;
    Theta_relative_yx((i-1)*n_valid_triggers_per_acquisition_cycle+1:((i)*n_valid_triggers_per_acquisition_cycle),:) = TR_yx;

    Position_matrix_tot(:,:,(i-1)*n_valid_triggers_per_acquisition_cycle+1:((i)*n_valid_triggers_per_acquisition_cycle)) = Position_matrix;

    
end

%% fix X matrix to make everything line up
% probably a modified version of get start of pulse function

[a,b] = size(X);

for j = 1:a 
    
    ind = find(X(j,1:2000) < spike_detect_thresh);
    position = ind(1);
    
    X(j,1:b-(position-1)) = X(j,position:end);
    
end


%%

% save waveform data Currently saves to the local directory cause im lazy 
save([fname,'X.mat'],'X');
save([fname,'Theta_relative_zx.mat'],'Theta_relative_zx');
save([fname,'Theta_relative_yx.mat'],'Theta_relative_yx');
save([fname,'Position_matrix_tot.mat'],'Position_matrix_tot');


% angle arrays
% save post_processing parameters
save([fname,'/post_processing_params.mat'],'post_processing_params');

%%

% plot if needed to verify the fit
%plotFitCurvatureImage(curvature_array,x_array,y_array,image_crop_array)



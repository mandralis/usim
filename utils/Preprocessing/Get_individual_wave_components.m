% Function that takes in the waveforms as features are added and outputs
% the component that each feature reflects. to be used with the inversion
% technique

function out_waves = Get_individual_wave_components(w,n_samples_per_waveform,spike_detect_thresh,noise_floor,zero_length)

%find out shape of w 

[a,b] = size(w);

if a > b == 1
    w = w';
    
end

position = get_start_of_pulse(w(1,:),n_samples_per_waveform,spike_detect_thresh,noise_floor,zero_length); 

crop_w = w(:,position:end); 



end
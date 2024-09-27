classdef AcquisitionParameters
    properties
        % data acquisition parameters
        n_acquisition_cycles_max          = 50;
        n_acquisition_cycles              = 50;
        t_per_acquisition                 = 2.5; % we are only actually triggering for 2 seconds but give 0.5 seconds buffer
        
        % data processing parameters
        n_samples_per_acquisition_cycle   = 5e6; % 5 M/s
        n_triggers_per_acquisition_cycle  = 400; % number of times camera triggers P/R (also the number of images we acquired)
    end
end
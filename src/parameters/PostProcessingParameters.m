classdef PostProcessingParameters
    properties
        % image processing parameters
        mask_threshold                    = 70;
        image_crop_array                  = [1 1079 140 1790];
        smooth_before_fit                 = false;
        
        %phase delay parameters 
        spike_detect_thresh               = -3;
        noise_floor                       = 0.15
        zero_length                       = 1000;
    end
end
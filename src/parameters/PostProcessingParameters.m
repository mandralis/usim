classdef PostProcessingParameters
    properties
        % image processing parameters
        mask_threshold                    = 110;
        image_crop_array                  = [1 1080 1 1920];
        smooth_before_fit                 = false;
        shadow_removal_array              = [904,1080,884,1052]
        wire_length                       = 1500;
        
        %phase delay parameters 
        spike_detect_thresh               = -3;
        noise_floor                       = 0.15
        zero_length                       = 1000;
    end
end
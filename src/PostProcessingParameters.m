classdef PostProcessingParameters
    properties
        % image processing parameters
        mask_threshold                    = 70;
        image_crop_array                  = [1 1079 140 1790];
        smooth_before_fit                 = false;
    end
end
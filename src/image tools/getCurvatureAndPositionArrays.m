function [curvature_array,x_array,y_array] = getCurvatureAndPositionArrays(datapath,mask_threshold,image_crop_array,smooth_before_fit)
    
    % dimensions of image after cropping
    nx = image_crop_array(4)-image_crop_array(3)+1;
    ny = image_crop_array(2)-image_crop_array(1)+1;

    % function handle producing the file path
    filepath = @ (i) [datapath, ['/Y7-S3 Camera',num2str(i,'%06.f'),'.tif']];
    i = 1;

    % count number of files
    while isfile(filepath(i))
        i = i+1;
    end
    N=i-1;

    % define curvature array
    curvature_array = zeros(N,nx);
    x_array         = zeros(N,nx);
    y_array         = zeros(N,nx);

    i=1;
    while isfile(filepath(i))
        disp(['Iteration: ',num2str(i)])

        % read image
        im = imread(filepath(i));

        % mask image
        im_masked = Grayim2mask(im,mask_threshold,image_crop_array);
    
        % get wire coordinates
        [xpoints,ypoints] = maskedimage2points(im_masked);

        % smooth before fit if necessary
        if smooth_before_fit
            ypoints = smooth(ypoints);
        end

        % fit a curve to wire coordinates
        [fitresult, ~]    = createFit(xpoints, ypoints);

        % sample curve for curvature calculation
        xpoints_fine = linspace(1,nx,nx);
        ypoints_fine = fitresult(xpoints_fine);     

        % compute curvature as a function of x
        [~,N,~,k,~] = frenet(xpoints_fine',ypoints_fine);

        % add curvature to curvature array
        curvature_array(i,:) = k.*-sign(N(:,2));

        % add positions to position arrays
        x_array(i,:) = xpoints_fine;
        y_array(i,:) = ypoints_fine;

        % increment loop
        i = i+1;   
    end
end

function [curvature_array,x_array,y_array] = getCurvatureAndPositionArrays(datapath,mask_threshold,image_crop_array,smooth_before_fit,shadow_removal_array,wire_length)
    
%     figure(1)
    % dimensions of image after cropping
    nx = image_crop_array(4)-image_crop_array(3)+1;
    ny = image_crop_array(2)-image_crop_array(1)+1;

    % function handle producing the file path
    filepath = @ (i) [datapath, ['Y7-S3 Camera',num2str(i,'%06.f'),'.tif']];
    i = 0;

    % count number of files
    while exist(filepath(i),'file') == 2
        i = i+1;
    end
    N=i-1;

    % define curvature array
    curvature_array = zeros(N,nx);
    x_array         = zeros(N,nx);
    y_array         = zeros(N,nx);

    i=1;
    while exist(filepath(i-1),'file') == 2
        
%         disp(['Iteration: ',num2str(i)])

        % read image
        im = imread(filepath(i-1));

        %remove shadows

        im(shadow_removal_array(1):shadow_removal_array(2),shadow_removal_array(3):shadow_removal_array(4)) = mask_threshold + 1;

        % mask image
        im_masked = Grayim2mask(im,mask_threshold,image_crop_array);
    
        % get wire coordinates
        [xpoints,ypoints] = maskedimage2points(im_masked);

        %remove nans
        xpoints(isnan(ypoints)) = [];
        ypoints(isnan(ypoints)) = [];

        % smooth before fit if necessary
        if smooth_before_fit
            ypoints = smooth(ypoints);
        end

        %trim everything after length calculation is done (get rid of my hand)
        dX = gradient(xpoints);
        dY = gradient(ypoints);
        Len = cumsum(hypot(dX,dY));

        xpoints(Len>wire_length) = [];
        ypoints(Len>wire_length) = [];

        % fit a curve to wire coordinates
        [fitresult, ~]    = createFit(xpoints, ypoints);

        % sample curve for curvature calculation
%         xpoints_fine = linspace(1,nx,nx); % this is assuming that the
%         wire leaves the frame of the camera on the right. but it doesnt 
        xpoints_fine = linspace(1,xpoints(end),xpoints(end)); 
        ypoints_fine = fitresult(xpoints_fine);     

        % compute curvature as a function of x
        [~,N,~,k,~] = frenet(xpoints_fine',ypoints_fine);

        % add curvature to curvature array
        curvature_array(i,1:xpoints(end)) = k.*-sign(N(:,2));

        % add positions to position arrays
        x_array(i,1:xpoints(end)) = xpoints_fine;
        y_array(i,1:xpoints(end)) = ypoints_fine;
       
        %display fit 
%         clf
%         imagesc(flipud(im))
%         hold on
%         scatter(xpoints_fine,ypoints_fine)
%         %scatter(xpoints,ypoints)
%         axis([1 1920 1 1080])
%         drawnow
%%         pause(0.01)


        % increment loop
        i = i+1;   
    end
%      close 1
end

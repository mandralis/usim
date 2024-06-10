
function plotFitCurvatureImage(curvature_array,x_array,y_array,image_crop_array)
% idx_x_plot = [100-10,100+10]; % plot curvature upstream
% dimensions of image after cropping
nx = image_crop_array(4)-image_crop_array(3)+1;
ny = image_crop_array(2)-image_crop_array(1)+1;

N = size(curvature_array,1);
figure();
for i = 1:N
    xpoints_fine = x_array(i,:);
    ypoints_fine = y_array(i,:);
    
    h0 = subplot(2,2,1);
    plot(xpoints_fine,ypoints_fine);
    axis(h0,[0 nx 0 ny])
    subplot(2,2,2)
    %imagesc(im_masked)
    
    h1 = subplot(2,2,3);
    plot(xpoints_fine,curvature_array);
    axis(h1,[0 nx -0.02 0.02])
    
    % h2 = subplot(2,2,4);
    % hold on
    % scatter(i,mean(k(idx_x_plot(1):idx_x_plot(2))))
    
    pause(0.1);
end

end


%
%         plot fitresult
%         plot(xpoints_fine,ypoints_fine);
%         axis([1,640,1,480]);
%         pause(1.0);

%% plot everything

%

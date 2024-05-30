function [xpoints,ypoints,outp] =Grayim2mask(im, thresh)

%im = rgb2gray(im);

im = double(im);

cropim = im(1:1079,130:1790);

threshim = cropim < thresh;

outp = threshim;

xpoints = 1;
ypoints = 1;
% %%
% 
% %generating points from the mask, we can set color preferance later
% 
% %setting region of interest from 399,152 to 2130,1258  
% 
% % cropped_pos = [399,152;2130,1258]; % for now assume no cropping is nescessary
% % 
% % cropped_im = test(152:1258,399:2130,:);
% % cropped_im = test(152:1258,399:2130,:);
% % cropped_mask = colormask(152:1258,399:2130);
% 
% cropped_mask = colormask;
% 
% %find where wire begins and ends
% 
% wire_beg = find(sum(cropped_mask(:,:) == wirecolor,1),1,'first');
% wire_end = find(sum(cropped_mask(:,:) == wirecolor,1),1,'last');
% 
% 
% % [a,b,~] = size(cropped_im);
% 
% counter = 1;
% 
% % find middle position of wire in y direction 
% for i = wire_beg:wire_end
%     f = find(cropped_mask(:,i),1,'first'); %find thickenss of wire
%     l = find(cropped_mask(:,i),1,'last');
%     
%     if isempty(l)
%         
%     else
%         y_pos(counter) = (f+l)/2; %find midpoint and save it to ouput point vecters
%         x_pos(counter) = i;
%         counter = counter+1;
%         % STILL NEED TO CORRECT FOR CROPPING
%         
%     end
% 
%     
% end
% 
% xpoints = x_pos;
% ypoints = y_pos;

end
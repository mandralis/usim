function [xpoints,ypoints,colormask] =image2points(im, wirecolor)

% this function will take in a image of a wire and return an array of points
% that lie on that wire. You can designate the color of the wire for eaisier 
% identification. The function will also return return the location of the 
% wire bending posts
% 
% [xpoints,ypoints] =image2points(im, wirecolor, postcolor)
% 
% im is the input color image 
% wirecolor is the color of the wire where:
% 1 = red
% 2 = green
% 3 = blue
% 

%%

%part one: find red, green and blue opbjects in the image

% im = double(test); %test images

im = double(im);

[a,b,~] = size(im); %size of image

light_thresh = 130; %luminance threshold (from 1:255)

double_thresh = 0.05; %percentage threshold for double colors

colormask = zeros(a,b);
%iterate through every pixel and assign color based on threshold
for i = 1:a
    for j = 1:b
        if im(i,j,1) > im(i,j,2) + im(i,j,3)
            colormask(i,j) = 1; %red
        elseif im(i,j,2) > im(i,j,1) + im(i,j,3)
            colormask(i,j) = 2; %blue
        elseif im(i,j,3) > im(i,j,1) + im(i,j,2)
            colormask(i,j) = 3; %green
        end
        
        %brightness estimate
        lumin = im(i,j,3) + im(i,j,1) + im(i,j,2); 
        
        if  light_thresh > lumin
            colormask(i,j) = 0; % ignore very dark pixels
            
            %this didnt work very well
%         elseif double_thresh > im(i,j,1)/(im(i,j,3) + im(i,j,1) + im(i,j,2))
%             colormask(i,j) = 0; % ignore double colors
%         elseif double_thresh > im(i,j,2)/(im(i,j,3) + im(i,j,1) + im(i,j,2))
%             colormask(i,j) = 0; % ignore double colors
%         elseif double_thresh > im(i,j,3)/(im(i,j,3) + im(i,j,1) + im(i,j,2))
%             colormask(i,j) = 0; % ignore double colors ?
        end
        
    end

end

%%

%generating points from the mask, we can set color preferance later

%setting region of interest from 399,152 to 2130,1258  

% cropped_pos = [399,152;2130,1258]; % for now assume no cropping is nescessary
% 
% cropped_im = test(152:1258,399:2130,:);
% cropped_im = test(152:1258,399:2130,:);
% cropped_mask = colormask(152:1258,399:2130);

cropped_mask = colormask;

%find where wire begins and ends

wire_beg = find(sum(cropped_mask(:,:) == wirecolor,1),1,'first');
wire_end = find(sum(cropped_mask(:,:) == wirecolor,1),1,'last');


% [a,b,~] = size(cropped_im);

counter = 1;

% find middle position of wire in y direction 
for i = wire_beg:wire_end
    f = find(cropped_mask(:,i),1,'first'); %find thickenss of wire
    l = find(cropped_mask(:,i),1,'last');
    
    if isempty(l)
        
    else
        y_pos(counter) = (f+l)/2; %find midpoint and save it to ouput point vecters
        x_pos(counter) = i;
        counter = counter+1;
        % STILL NEED TO CORRECT FOR CROPPING
        
    end

    
end

xpoints = x_pos;
ypoints = y_pos;

end
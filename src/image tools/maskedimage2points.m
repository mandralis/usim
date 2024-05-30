function [xpoints,ypoints] = maskedimage2points(cdata)
nx = size(cdata,2);
ny = size(cdata,1);
xpoints = 1:nx;
ypoints = [];
for x = xpoints
    wire_points = ny - mean(find(cdata(:,x) == 1));
    if isempty(wire_points)
    else
        ypoints = [ypoints wire_points];
    end
end
% figure();
% plot(xpoints,ypoints);
% axis([1,nx,1,ny]);
end
function outim =Grayim2mask(im, thresh, crop_array)
%%
%im = rgb2gray(im);

im = double(im);

% cropim = im; 
cropim = im(crop_array(1):crop_array(2),crop_array(3):crop_array(4));

threshim = cropim < thresh;

outim = threshim;


end
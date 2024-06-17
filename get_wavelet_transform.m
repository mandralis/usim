% clear workspace
% clear all
% close all
% clc

% add necessary paths
addpath(genpath('src/'));
addpath(genpath('utils/'));

data_path = 'data_05_26_2024_16_54_50/';
addpath(genpath([get_local_data_path(),data_path]));
load([data_path,'acquisition_params.mat']);
load([data_path,'X.mat']);


%time position of sensors being used
sensor_indeces = [1308, 1395, 1488,1584];


[a,b] = size(X);

%find amplitude components of these sensors
for i = 1:a
   
    i
    spect = cwt(X(i,1:2000),2500000);
    
    for j = 1:4
        ampl(i,j) = abs(spect(20,sensor_indeces(j)));
        
    end
       

end

%%
save('ampl.mat','ampl');

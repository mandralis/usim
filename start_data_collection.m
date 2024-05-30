% clear workspace
clear all
close all
clc

% instantiate object from Parameters class
acquisition_params = AcquisitionParameters();

% connect matlab to oscilloscope 
myScope = connect();

% connect to arduino
a = arduino();
writeDigitalPin(a,'D4',0); %make sure pin is 0
pause(2);

% create a time stamped data folder to save the waveform + wire shape in
fname = new_folder('data');

% put oscilloscope in right mode by acquiring waveform once
get_waveform(myScope);

% now start the data acquisition loop
for i = 1:acquisition_params.n_acquisition_cycles

    % trigger arduino to analog trigger camera
    trigger_arduino(a);  

    % wait for acquisition time
    pause(acquisition_params.t_per_acquisition);

    % save waveform to computer
    w = get_waveform(myScope);
    save([fname,'/pulse_',num2str(i),'.mat'],'w');

end

% save parameters used for loading the data
save([fname,'/acquisition_params.mat'],'acquisition_params');

% disconnect oscilloscope
disconnect(myScope)
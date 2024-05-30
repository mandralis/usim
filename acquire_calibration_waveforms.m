%used to get calibaration waveforms for the wire. First waveform is the
%bare wire, then a surface feature is added for each subsequent waveform.

% instantiate object from Parameters class
params = Parameters();

% connect matlab to oscilloscope 
myScope = connect();

count = 1;
while 1 == 1 
    disp(count);
    w(:,count) = getWaveform(myScope, 'acquisition', true);
    pause
    count = count +1;
end

% save calibration waveform to path
% create a time stamped data folder to save the waveform + wire shape in
fname = new_folder('calibration');
save([fname,'/calibration.mat'],'w');
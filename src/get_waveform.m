function w = get_waveform(myScope)
    myScope.SingleSweepMode = 'off';
    w = getWaveform(myScope, 'acquisition', true); % useless data 
    myScope.SingleSweepMode = 'on';
end
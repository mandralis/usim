function position = get_start_of_pulse(w,n_samples_per_waveform,spike_detect_thresh,noise_floor,zero_length)

%with the two waveforms loaded, find excitation spike 

%first check that signla is zero for a long time. then find first spike
%above a threshhold... probably around 0.4 ish. 

%%
%Parameters, make sure that they same as param file 


%find signal spike below thresh cause negative spike

waveform  = w;

zerowaveform = waveform;
zerowaveform(1:zero_length+1) = 0;

ind = find(zerowaveform < spike_detect_thresh);

%check for a long enough time before spike of zero signal 
flag = 1;
i = 0; %counter

while flag == 1
    i = i+1;
    
    if i == length(waveform) 
        disp('no peaks detected')
        flag = 0;
        position = nan;
    elseif sum(abs(waveform(ind(i)-(zero_length-1):ind(i)-1)) > noise_floor) == 0
        
        flag = 0;
        position = ind(i);
    end
    
    
end

if position > n_samples_per_waveform
    ind = find(waveform < spike_detect_thresh);
    position = ind(1);
end
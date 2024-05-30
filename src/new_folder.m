function fname = new_folder(name)
    % create a time stamped data folder to save the waveform + wire shape in
    time = datetime('now');
    time.Format = 'MM_dd_yyyy_HH_mm_ss';
    fname = char(time);
    fname = [name,'_',fname];
    mkdir(fname);
end
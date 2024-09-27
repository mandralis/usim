function fname = new_folder(name)
    % get local data path
    local_data_path = get_local_data_path();

    % create a time stamped data folder to save the waveform + wire shape in
    time = datetime('now');
    time.Format = 'MM_dd_yyyy_HH_mm_ss';
    fname = char(time);
    fname = [name,'_',fname];
    fname = [local_data_path,fname];
    mkdir(fname);
end
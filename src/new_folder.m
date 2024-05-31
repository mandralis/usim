function fname = new_folder(name)
    % get local data path
    data_path = get_data_path();

    % create a time stamped data folder to save the waveform + wire shape in
    time = datetime('now');
    time.Format = 'MM_dd_yyyy_HH_mm_ss';
    fname = char(time);
    fname = [name,'_',fname];
    mkdir(home_path + fname);
end
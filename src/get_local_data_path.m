function data_path = get_local_data_path()
    if ispc
        data_path = 'C:\Users\arosa\Box\USS Catheter\data\';
    else
        data_path = '/Users/imandralis/box/USS Catheter/data/';
end
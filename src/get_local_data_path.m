function data_path = get_local_data_path()
    if ispc
        data_path = 'C:\Users\arosa\Box\USS Catheter\data\'; % use for 3d: 'C:\Users\arosa\Box\USS Catheter\3d_data\'
    else
        data_path = '/Users/imandralis/box/USS Catheter/data/';

    end

end
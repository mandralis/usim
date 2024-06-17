% Import the ONNX model as a dlnetwork
net = importONNXNetwork('model.onnx', 'InputDataFormats', {'BC'}, 'TargetNetwork', 'dlnetwork');

% Example input data in the 'BC' format (batch size 1, 2000 channels)
inputData = rand(1, 2000);

% Convert input data to a dlarray
dlInput = dlarray(inputData, 'BC');

% Perform inference
output = predict(net, dlInput);

% Display the output
disp(output);
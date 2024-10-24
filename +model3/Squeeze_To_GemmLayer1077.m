classdef Squeeze_To_GemmLayer1077 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    
    %#codegen
    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    
    properties (Learnable)
        bn_layers_4_bias
        bn_layers_4_weight
        fc_bias
        fc_weight
    end
    
    properties
        ONNXParams         % An ONNXParameters object containing parameters used by this layer.
    end
    
    methods
        function this = Squeeze_To_GemmLayer1077(name, onnxParams)
            this.Name = name;
            this.OutputNames = {'output'};
            this.ONNXParams = onnxParams;
            this.bn_layers_4_bias = onnxParams.Learnables.bn_layers_4_bias;
            this.bn_layers_4_weight = onnxParams.Learnables.bn_layers_4_weight;
            this.fc_bias = onnxParams.Learnables.fc_bias;
            this.fc_weight = onnxParams.Learnables.fc_weight;
        end
        
        function [output] = predict(this, x_conv_layers_4_Conv)
            if isdlarray(x_conv_layers_4_Conv)
                x_conv_layers_4_Conv = stripdims(x_conv_layers_4_Conv);
            end
            x_conv_layers_4_ConvNumDims = 3;
            onnxParams = this.ONNXParams;
            onnxParams.Learnables.bn_layers_4_bias = this.bn_layers_4_bias;
            onnxParams.Learnables.bn_layers_4_weight = this.bn_layers_4_weight;
            onnxParams.Learnables.fc_bias = this.fc_bias;
            onnxParams.Learnables.fc_weight = this.fc_weight;
            [output, outputNumDims] = Squeeze_To_GemmFcn(x_conv_layers_4_Conv, x_conv_layers_4_ConvNumDims, onnxParams, 'Training', false, ...
                'InputDataPermutation', {[2 1 3], ['as-is']}, ...
                'OutputDataPermutation', {['as-is'], ['as-is']});
            if any(cellfun(@(A)isempty(A)||~isnumeric(A), {output}))
                fprintf('Runtime error in network. The custom layer ''%s'' output an empty or non-numeric value.\n', 'Squeeze_To_GemmLayer1077');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Squeeze_To_GemmLayer1077'));
            end
            output = dlarray(single(output), repmat('U', 1, max(2, outputNumDims)));
            if ~coder.target('MATLAB')
                output = extractdata(output);
            end
        end
        
        function [output] = forward(this, x_conv_layers_4_Conv)
            if isdlarray(x_conv_layers_4_Conv)
                x_conv_layers_4_Conv = stripdims(x_conv_layers_4_Conv);
            end
            x_conv_layers_4_ConvNumDims = 3;
            onnxParams = this.ONNXParams;
            onnxParams.Learnables.bn_layers_4_bias = this.bn_layers_4_bias;
            onnxParams.Learnables.bn_layers_4_weight = this.bn_layers_4_weight;
            onnxParams.Learnables.fc_bias = this.fc_bias;
            onnxParams.Learnables.fc_weight = this.fc_weight;
            [output, outputNumDims] = Squeeze_To_GemmFcn(x_conv_layers_4_Conv, x_conv_layers_4_ConvNumDims, onnxParams, 'Training', true, ...
                'InputDataPermutation', {[2 1 3], ['as-is']}, ...
                'OutputDataPermutation', {['as-is'], ['as-is']});
            if any(cellfun(@(A)isempty(A)||~isnumeric(A), {output}))
                fprintf('Runtime error in network. The custom layer ''%s'' output an empty or non-numeric value.\n', 'Squeeze_To_GemmLayer1077');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Squeeze_To_GemmLayer1077'));
            end
            output = dlarray(single(output), repmat('U', 1, max(2, outputNumDims)));
            if ~coder.target('MATLAB')
                output = extractdata(output);
            end
        end
    end
end

function [output, outputNumDims, state] = Squeeze_To_GemmFcn(x_conv_layers_4_Conv, x_conv_layers_4_ConvNumDims, params, varargin)
%SQUEEZE_TO_GEMMFCN Function implementing an imported ONNX network.
%
% THIS FILE WAS AUTO-GENERATED BY importONNXFunction.
% ONNX Operator Set Version: 11
%
% Variable names in this function are taken from the original ONNX file.
%
% [OUTPUT] = Squeeze_To_GemmFcn(X_CONV_LAYERS_4_CONV, PARAMS)
%			- Evaluates the imported ONNX network SQUEEZE_TO_GEMMFCN with input(s)
%			X_CONV_LAYERS_4_CONV and the imported network parameters in PARAMS. Returns
%			network output(s) in OUTPUT.
%
% [OUTPUT, STATE] = Squeeze_To_GemmFcn(X_CONV_LAYERS_4_CONV, PARAMS)
%			- Additionally returns state variables in STATE. When training,
%			use this form and set TRAINING to true.
%
% [__] = Squeeze_To_GemmFcn(X_CONV_LAYERS_4_CONV, PARAMS, 'NAME1', VAL1, 'NAME2', VAL2, ...)
%			- Specifies additional name-value pairs described below:
%
% 'Training'
% 			Boolean indicating whether the network is being evaluated for
%			prediction or training. If TRAINING is true, state variables
%			will be updated.
%
% 'InputDataPermutation'
%			'auto' - Automatically attempt to determine the permutation
%			 between the dimensions of the input data and the dimensions of
%			the ONNX model input. For example, the permutation from HWCN
%			(MATLAB standard) to NCHW (ONNX standard) uses the vector
%			[4 3 1 2]. See the documentation for IMPORTONNXFUNCTION for
%			more information about automatic permutation.
%
%			'none' - Input(s) are passed in the ONNX model format. See 'Inputs'.
%
%			numeric vector - The permutation vector describing the
%			transformation between input data dimensions and the expected
%			ONNX input dimensions.%
%			cell array - If the network has multiple inputs, each cell
%			contains 'auto', 'none', or a numeric vector.
%
% 'OutputDataPermutation'
%			'auto' - Automatically attempt to determine the permutation
%			between the dimensions of the output and a conventional MATLAB
%			dimension ordering. For example, the permutation from NC (ONNX
%			standard) to CN (MATLAB standard) uses the vector [2 1]. See
%			the documentation for IMPORTONNXFUNCTION for more information
%			about automatic permutation.
%
%			'none' - Return output(s) as given by the ONNX model. See 'Outputs'.
%
%			numeric vector - The permutation vector describing the
%			transformation between the ONNX output dimensions and the
%			desired output dimensions.%
%			cell array - If the network has multiple outputs, each cell
%			contains 'auto', 'none' or a numeric vector.
%
% Inputs:
% -------
% X_CONV_LAYERS_4_CONV
%			- Input(s) to the ONNX network.
%			  The input size(s) expected by the ONNX file are:
%				  X_CONV_LAYERS_4_CONV:		[Unknown, Unknown, Unknown]				Type: FLOAT
%			  By default, the function will try to permute the input(s)
%			  into this dimension ordering. If the default is incorrect,
%			  use the 'InputDataPermutation' argument to control the
%			  permutation.
%
%
% PARAMS	- Network parameters returned by 'importONNXFunction'.
%
%
% Outputs:
% --------
% OUTPUT
%			- Output(s) of the ONNX network.
%			  Without permutation, the size(s) of the outputs are:
%				  OUTPUT:		[1, 9]				Type: FLOAT
%			  By default, the function will try to permute the output(s)
%			  from this dimension ordering into a conventional MATLAB
%			  ordering. If the default is incorrect, use the
%			  'OutputDataPermutation' argument to control the permutation.
%
% STATE		- (Optional) State variables. When TRAINING is true, these will
% 			  have been updated from the original values in PARAMS.State.
%
%
%  See also importONNXFunction

% Preprocess the input data and arguments:
[x_conv_layers_4_Conv, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(x_conv_layers_4_Conv, params, varargin{:});
% Put all variables into a single struct to implement dynamic scoping:
[Vars, NumDims] = packageVariables(params, {'x_conv_layers_4_Conv'}, {x_conv_layers_4_Conv}, [x_conv_layers_4_ConvNumDims]);
% Call the top-level graph function:
[output, outputNumDims, state] = Squeeze_To_GemmGraph1070(x_conv_layers_4_Conv, NumDims.x_conv_layers_4_Conv, Vars, NumDims, Training, params.State);
% Postprocess the output data
[output] = postprocessOutput(output, outputDataPerms, anyDlarrayInputs, Training, varargin{:});
end

function [output, outputNumDims1076, state] = Squeeze_To_GemmGraph1070(x_conv_layers_4_Conv, x_conv_layers_4_ConvNumDims1075, Vars, NumDims, Training, state)
% Function implementing the graph 'Squeeze_To_GemmGraph1070'
% Update Vars and NumDims from the graph's formal input parameters. Note that state variables are already in Vars.
Vars.x_conv_layers_4_Conv = x_conv_layers_4_Conv;
NumDims.x_conv_layers_4_Conv = x_conv_layers_4_ConvNumDims1075;

% Execute the operators:
% Squeeze:
[Vars.x_conv_layers_4_Sque, NumDims.x_conv_layers_4_Sque] = onnxSqueeze(Vars.x_conv_layers_4_Conv, Vars.SqueezeAxes1071, NumDims.x_conv_layers_4_Conv);

% Transpose:
[perm, NumDims.x_Transpose_9_output] = prepareTransposeArgs(Vars.TransposePerm1072, NumDims.x_conv_layers_4_Sque);
if ~isempty(perm)
    Vars.x_Transpose_9_output = permute(Vars.x_conv_layers_4_Sque, perm);
end

% BatchNormalization:
[offset, scale, datasetMean, datasetVariance, dataFormat, NumDims.x_bn_layers_4_BatchN, NumDims.bn_layers_4_running_, NumDims.bn_layers_4_runnin_1] = prepareBatchNormalizationArgs(Vars.bn_layers_4_bias, Vars.bn_layers_4_weight, Vars.bn_layers_4_running_, Vars.bn_layers_4_runnin_1, NumDims.x_Transpose_9_output, NumDims.bn_layers_4_running_, NumDims.bn_layers_4_runnin_1);
if Training
    [Vars.x_bn_layers_4_BatchN, dsmean, dsvar] = batchnorm(Vars.x_Transpose_9_output, offset, scale, datasetMean, datasetVariance, 'Epsilon', 0.000010, 'DataFormat', dataFormat);
    Vars.bn_layers_4_running_ = dlarray(dsmean);
    Vars.bn_layers_4_runnin_1 = dlarray(dsvar);
else
    Vars.x_bn_layers_4_BatchN = batchnorm(Vars.x_Transpose_9_output, offset, scale, datasetMean, datasetVariance, 'Epsilon', 0.000010, 'DataFormat', dataFormat);
end
state.bn_layers_4_running_ = Vars.bn_layers_4_running_;
state.bn_layers_4_runnin_1 = Vars.bn_layers_4_runnin_1;

% Relu:
Vars.x_relu_layers_4_Relu = relu(Vars.x_bn_layers_4_BatchN);
NumDims.x_relu_layers_4_Relu = NumDims.x_bn_layers_4_BatchN;

% Gemm:
[A, B, C, alpha, beta, NumDims.output] = prepareGemmArgs(Vars.x_relu_layers_4_Relu, Vars.fc_weight, Vars.fc_bias, Vars.Gemmalpha1073, Vars.Gemmbeta1074, 0, 1, NumDims.fc_bias);
Vars.output = alpha*B*A + beta*C;

% Set graph output arguments from Vars and NumDims:
output = Vars.output;
outputNumDims1076 = NumDims.output;
% Set output state from Vars:
state = updateStruct(state, Vars);
end

function [inputDataPerms, outputDataPerms, Training] = parseInputs(x_conv_layers_4_Conv, numDataOutputs, params, varargin)
% Function to validate inputs to Squeeze_To_GemmFcn:
p = inputParser;
isValidArrayInput = @(x)isnumeric(x) || isstring(x);
isValidONNXParameters = @(x)isa(x, 'ONNXParameters');
addRequired(p, 'x_conv_layers_4_Conv', isValidArrayInput);
addRequired(p, 'params', isValidONNXParameters);
addParameter(p, 'InputDataPermutation', 'auto');
addParameter(p, 'OutputDataPermutation', 'auto');
addParameter(p, 'Training', false);
parse(p, x_conv_layers_4_Conv, params, varargin{:});
inputDataPerms = p.Results.InputDataPermutation;
outputDataPerms = p.Results.OutputDataPermutation;
Training = p.Results.Training;
if isnumeric(inputDataPerms)
    inputDataPerms = {inputDataPerms};
end
if isstring(inputDataPerms) && isscalar(inputDataPerms) || ischar(inputDataPerms)
    inputDataPerms = repmat({inputDataPerms},1,1);
end
if isnumeric(outputDataPerms)
    outputDataPerms = {outputDataPerms};
end
if isstring(outputDataPerms) && isscalar(outputDataPerms) || ischar(outputDataPerms)
    outputDataPerms = repmat({outputDataPerms},1,numDataOutputs);
end
end

function [x_conv_layers_4_Conv, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(x_conv_layers_4_Conv, params, varargin)
% Parse input arguments
[inputDataPerms, outputDataPerms, Training] = parseInputs(x_conv_layers_4_Conv, 1, params, varargin{:});
anyDlarrayInputs = any(cellfun(@(x)isa(x, 'dlarray'), {x_conv_layers_4_Conv}));
% Make the input variables into unlabelled dlarrays:
x_conv_layers_4_Conv = makeUnlabeledDlarray(x_conv_layers_4_Conv);
% Permute inputs if requested:
x_conv_layers_4_Conv = permuteInputVar(x_conv_layers_4_Conv, inputDataPerms{1}, 3);
end

function [output] = postprocessOutput(output, outputDataPerms, anyDlarrayInputs, Training, varargin)
% Set output type:
if ~anyDlarrayInputs && ~Training
    if isdlarray(output)
        output = extractdata(output);
    end
end
% Permute outputs if requested:
output = permuteOutputVar(output, outputDataPerms{1}, 2);
end


%% dlarray functions implementing ONNX operators:

function [Y, numDimsY] = onnxSqueeze(X, ONNXAxes, numDimsX)
% Implements the ONNX Squeeze operator
if numDimsX == 0
    Y = X;
    numDimsY = numDimsX;
else
    % Find the new ONNX shape
    curOShape = size(X, numDimsX:-1:1);
    if isempty(ONNXAxes)
        newOShape = curOShape(curOShape ~= 1);
    else
        ONNXAxes(ONNXAxes<0) = ONNXAxes(ONNXAxes<0) + numDimsX;
        newOShape = curOShape;
        newOShape(ONNXAxes+1) = [];
    end
    % Get numDimsY from ONNX shape
    numDimsY  = numel(newOShape);
    newMShape = [fliplr(newOShape) ones(1, 2-numDimsY)];    % Append 1's to shape if numDims<2
    Y         = reshape(X, newMShape);
end
end

function [offset, scale, datasetMean, datasetVariance, dataFormat, numDimsY, numDimsDatasetMean, numDimsDatasetVariance] = prepareBatchNormalizationArgs(...
    offset, scale, datasetMean, datasetVariance, numDimsX, numDimsDatasetMean, numDimsDatasetVariance)
% Prepares arguments for implementing the ONNX BatchNormalization operator
offset = dlarray(offset,'C');
scale = dlarray(scale,'C');
datasetMean = extractdata(datasetMean);
datasetVariance = extractdata(datasetVariance);
datasetVariance(datasetVariance <= 0) = realmin('single');  % Set nonpositive variance components to a value below eps('single')
dataFormat = [repmat('S', 1, numDimsX-2), 'CB'];
numDimsY = numDimsX;
end

function [A, B, C, alpha, beta, numDimsY] = prepareGemmArgs(A, B, C, alpha, beta, transA, transB, numDimsC)
% Prepares arguments for implementing the ONNX Gemm operator
if transA
    A = A';
end
if transB
    B = B';
end
if numDimsC < 2
    C = C(:);   % C can be broadcast to [N M]. Make C a col vector ([N 1])
end
numDimsY = 2;
% Y=B*A because we want (AB)'=B'A', and B and A are already transposed.
end

function [perm, numDimsA] = prepareTransposeArgs(ONNXPerm, numDimsA)
% Prepares arguments for implementing the ONNX Transpose operator
if numDimsA <= 1        % Tensors of numDims 0 or 1 are unchanged by ONNX Transpose.
    perm = [];
else
    if isempty(ONNXPerm)        % Empty ONNXPerm means reverse the dimensions.
        perm = numDimsA:-1:1;
    else
        perm = numDimsA-flip(ONNXPerm);
    end
end
end

%% Utility functions:

function s = appendStructs(varargin)
% s = appendStructs(s1, s2,...). Assign all fields in s1, s2,... into s.
if isempty(varargin)
    s = struct;
else
    s = varargin{1};
    for i = 2:numel(varargin)
        fromstr = varargin{i};
        fs = fieldnames(fromstr);
        for j = 1:numel(fs)
            s.(fs{j}) = fromstr.(fs{j});
        end
    end
end
end

function checkInputSize(inputShape, expectedShape, inputName)

if numel(expectedShape)==0
    % The input is a scalar
    if ~isequal(inputShape, [1 1])
        inputSizeStr = makeSizeString(inputShape);
        error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, "[1,1]", inputSizeStr));
    end
elseif numel(expectedShape)==1
    % The input is a vector
    if ~shapeIsColumnVector(inputShape) || ~iSizesMatch({inputShape(1)}, expectedShape)
        expectedShape{2} = 1;
        expectedSizeStr = makeSizeString(expectedShape);
        inputSizeStr = makeSizeString(inputShape);
        error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, expectedSizeStr, inputSizeStr));
    end
else
    % The input has 2 dimensions or more
    
    % The input dimensions have been reversed; flip them back to compare to the
    % expected ONNX shape.
    inputShape = fliplr(inputShape);
    
    % If the expected shape has fewer dims than the input shape, error.
    if numel(expectedShape) < numel(inputShape)
        expectedSizeStr = strjoin(["[", strjoin(string(expectedShape), ","), "]"], "");
        error(message('nnet_cnn_onnx:onnx:InputHasGreaterNDims', inputName, expectedSizeStr));
    end
    
    % Prepad the input shape with trailing ones up to the number of elements in
    % expectedShape
    inputShape = num2cell([ones(1, numel(expectedShape) - length(inputShape)) inputShape]);
    
    % Find the number of variable size dimensions in the expected shape
    numVariableInputs = sum(cellfun(@(x) isa(x, 'char') || isa(x, 'string'), expectedShape));
    
    % Find the number of input dimensions that are not in the expected shape
    % and cannot be represented by a variable dimension
    nonMatchingInputDims = setdiff(string(inputShape), string(expectedShape));
    numNonMatchingInputDims  = numel(nonMatchingInputDims) - numVariableInputs;
    
    expectedSizeStr = makeSizeString(expectedShape);
    inputSizeStr = makeSizeString(inputShape);
    if numNonMatchingInputDims == 0 && ~iSizesMatch(inputShape, expectedShape)
        % The actual and expected input dimensions match, but in
        % a different order. The input needs to be permuted.
        error(message('nnet_cnn_onnx:onnx:InputNeedsPermute',inputName, expectedSizeStr, inputSizeStr));
    elseif numNonMatchingInputDims > 0
        % The actual and expected input sizes do not match.
        error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, expectedSizeStr, inputSizeStr));
    end
end
end

function doesMatch = iSizesMatch(inputShape, expectedShape)
% Check whether the input and expected shapes match, in order.
% Size elements match if (1) the elements are equal, or (2) the expected
% size element is a variable (represented by a character vector or string)
doesMatch = true;
for i=1:numel(inputShape)
    if ~(isequal(inputShape{i},expectedShape{i}) || ischar(expectedShape{i}) || isstring(expectedShape{i}))
        doesMatch = false;
        return
    end
end
end

function sizeStr = makeSizeString(shape)
sizeStr = strjoin(["[", strjoin(string(shape), ","), "]"], "");
end

function isVec = shapeIsColumnVector(shape)
if numel(shape) == 2 && shape(2) == 1
    isVec = true;
else
    isVec = false;
end
end
function X = makeUnlabeledDlarray(X)
% Make numeric X into an unlabelled dlarray
if isa(X, 'dlarray')
    X = stripdims(X);
elseif isnumeric(X)
    if isinteger(X)
        % Make ints double so they can combine with anything without
        % reducing precision
        X = double(X);
    end
    X = dlarray(X);
end
end

function [Vars, NumDims] = packageVariables(params, inputNames, inputValues, inputNumDims)
% inputNames, inputValues are cell arrays. inputRanks is a numeric vector.
Vars = appendStructs(params.Learnables, params.Nonlearnables, params.State);
NumDims = params.NumDimensions;
% Add graph inputs
for i = 1:numel(inputNames)
    Vars.(inputNames{i}) = inputValues{i};
    NumDims.(inputNames{i}) = inputNumDims(i);
end
end

function X = permuteInputVar(X, userDataPerm, onnxNDims)
% Returns reverse-ONNX ordering
if onnxNDims == 0
    return;
elseif onnxNDims == 1 && isvector(X)
    X = X(:);
    return;
elseif isnumeric(userDataPerm)
    % Permute into reverse ONNX ordering
    if numel(userDataPerm) ~= onnxNDims
        error(message('nnet_cnn_onnx:onnx:InputPermutationSize', numel(userDataPerm), onnxNDims));
    end
    perm = fliplr(userDataPerm);
elseif isequal(userDataPerm, 'auto') && onnxNDims == 4
    % Permute MATLAB HWCN to reverse onnx (WHCN)
    perm = [2 1 3 4];
elseif isequal(userDataPerm, 'as-is')
    % Do not permute the input
    perm = 1:ndims(X);
else
    % userDataPerm is either 'none' or 'auto' with no default, which means
    % it's already in onnx ordering, so just make it reverse onnx
    perm = max(2,onnxNDims):-1:1;
end
X = permute(X, perm);
end

function Y = permuteOutputVar(Y, userDataPerm, onnxNDims)
switch onnxNDims
    case 0
        perm = [];
    case 1
        if isnumeric(userDataPerm)
            % Use the user's permutation because Y is a column vector which
            % already matches ONNX.
            perm = userDataPerm;
        elseif isequal(userDataPerm, 'auto')
            % Treat the 1D onnx vector as a 2D column and transpose it
            perm = [2 1];
        else
            % userDataPerm is 'none'. Leave Y alone because it already
            % matches onnx.
            perm = [];
        end
    otherwise
        % ndims >= 2
        if isnumeric(userDataPerm)
            % Use the inverse of the user's permutation. This is not just the
            % flip of the permutation vector.
            perm = onnxNDims + 1 - userDataPerm;
        elseif isequal(userDataPerm, 'auto')
            if onnxNDims == 2
                % Permute reverse ONNX CN to DLT CN (do nothing)
                perm = [];
            elseif onnxNDims == 4
                % Permute reverse onnx (WHCN) to MATLAB HWCN
                perm = [2 1 3 4];
            else
                % User wants the output in ONNX ordering, so just reverse it from
                % reverse onnx
                perm = onnxNDims:-1:1;
            end
        elseif isequal(userDataPerm, 'as-is')
            % Do not permute the input
            perm = 1:ndims(Y);
        else
            % userDataPerm is 'none', so just make it reverse onnx
            perm = onnxNDims:-1:1;
        end
end
if ~isempty(perm)
    Y = permute(Y, perm);
end
end

function s = updateStruct(s, t)
% Set all existing fields in s from fields in t, ignoring extra fields in t.
for name = transpose(fieldnames(s))
    s.(name{1}) = t.(name{1});
end
end

classdef Transpose_To_UnsqueezeLayer1025 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    
    %#codegen
    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    
    properties (Learnable)
    end
    
    properties
        ONNXParams         % An ONNXParameters object containing parameters used by this layer.
    end
    
    methods
        function this = Transpose_To_UnsqueezeLayer1025(name, onnxParams)
            this.Name = name;
            this.OutputNames = {'x_conv_layers_0_Unsq'};
            this.ONNXParams = onnxParams;
        end
        
        function [x_conv_layers_0_Unsq] = predict(this, input)
            if isdlarray(input)
                input = stripdims(input);
            end
            inputNumDims = 2;
            onnxParams = this.ONNXParams;
            [x_conv_layers_0_Unsq, x_conv_layers_0_UnsqNumDims] = Transpose_To_UnsqueezeFcn(input, inputNumDims, onnxParams, 'Training', false, ...
                'InputDataPermutation', {[2 1], ['as-is']}, ...
                'OutputDataPermutation', {[2 1 3], ['as-is']});
            if any(cellfun(@(A)isempty(A)||~isnumeric(A), {x_conv_layers_0_Unsq}))
                fprintf('Runtime error in network. The custom layer ''%s'' output an empty or non-numeric value.\n', 'Transpose_To_UnsqueezeLayer1025');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Transpose_To_UnsqueezeLayer1025'));
            end
            x_conv_layers_0_Unsq = dlarray(single(x_conv_layers_0_Unsq), 'CBT');
            if ~coder.target('MATLAB')
                x_conv_layers_0_Unsq = extractdata(x_conv_layers_0_Unsq);
            end
        end
        
        function [x_conv_layers_0_Unsq] = forward(this, input)
            if isdlarray(input)
                input = stripdims(input);
            end
            inputNumDims = 2;
            onnxParams = this.ONNXParams;
            [x_conv_layers_0_Unsq, x_conv_layers_0_UnsqNumDims] = Transpose_To_UnsqueezeFcn(input, inputNumDims, onnxParams, 'Training', true, ...
                'InputDataPermutation', {[2 1], ['as-is']}, ...
                'OutputDataPermutation', {[2 1 3], ['as-is']});
            if any(cellfun(@(A)isempty(A)||~isnumeric(A), {x_conv_layers_0_Unsq}))
                fprintf('Runtime error in network. The custom layer ''%s'' output an empty or non-numeric value.\n', 'Transpose_To_UnsqueezeLayer1025');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Transpose_To_UnsqueezeLayer1025'));
            end
            x_conv_layers_0_Unsq = dlarray(single(x_conv_layers_0_Unsq), 'CBT');
            if ~coder.target('MATLAB')
                x_conv_layers_0_Unsq = extractdata(x_conv_layers_0_Unsq);
            end
        end
    end
end

function [x_conv_layers_0_Unsq, x_conv_layers_0_UnsqNumDims, state] = Transpose_To_UnsqueezeFcn(input, inputNumDims, params, varargin)
%TRANSPOSE_TO_UNSQUEEZEFCN Function implementing an imported ONNX network.
%
% THIS FILE WAS AUTO-GENERATED BY importONNXFunction.
% ONNX Operator Set Version: 11
%
% Variable names in this function are taken from the original ONNX file.
%
% [X_CONV_LAYERS_0_UNSQ] = Transpose_To_UnsqueezeFcn(INPUT, PARAMS)
%			- Evaluates the imported ONNX network TRANSPOSE_TO_UNSQUEEZEFCN with input(s)
%			INPUT and the imported network parameters in PARAMS. Returns
%			network output(s) in X_CONV_LAYERS_0_UNSQ.
%
% [X_CONV_LAYERS_0_UNSQ, STATE] = Transpose_To_UnsqueezeFcn(INPUT, PARAMS)
%			- Additionally returns state variables in STATE. When training,
%			use this form and set TRAINING to true.
%
% [__] = Transpose_To_UnsqueezeFcn(INPUT, PARAMS, 'NAME1', VAL1, 'NAME2', VAL2, ...)
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
% INPUT
%			- Input(s) to the ONNX network.
%			  The input size(s) expected by the ONNX file are:
%				  INPUT:		[1, 1000]				Type: FLOAT
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
% X_CONV_LAYERS_0_UNSQ
%			- Output(s) of the ONNX network.
%			  Without permutation, the size(s) of the outputs are:
%				  X_CONV_LAYERS_0_UNSQ:		[Unknown, Unknown, Unknown]				Type: FLOAT
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
[input, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(input, params, varargin{:});
% Put all variables into a single struct to implement dynamic scoping:
[Vars, NumDims] = packageVariables(params, {'input'}, {input}, [inputNumDims]);
% Call the top-level graph function:
[x_conv_layers_0_Unsq, x_conv_layers_0_UnsqNumDims, state] = Transpose_To_UnsqueeGraph1020(input, NumDims.input, Vars, NumDims, Training, params.State);
% Postprocess the output data
[x_conv_layers_0_Unsq] = postprocessOutput(x_conv_layers_0_Unsq, outputDataPerms, anyDlarrayInputs, Training, varargin{:});
end

function [x_conv_layers_0_Unsq, x_conv_layers_0_UnsqNumDims1024, state] = Transpose_To_UnsqueeGraph1020(input, inputNumDims1023, Vars, NumDims, Training, state)
% Function implementing the graph 'Transpose_To_UnsqueeGraph1020'
% Update Vars and NumDims from the graph's formal input parameters. Note that state variables are already in Vars.
Vars.input = input;
NumDims.input = inputNumDims1023;

% Execute the operators:
% Transpose:
[perm, NumDims.x_Transpose_output_0] = prepareTransposeArgs(Vars.TransposePerm1021, NumDims.input);
if ~isempty(perm)
    Vars.x_Transpose_output_0 = permute(Vars.input, perm);
end

% Unsqueeze:
[shape, NumDims.x_conv_layers_0_Unsq] = prepareUnsqueezeArgs(Vars.x_Transpose_output_0, Vars.UnsqueezeAxes1022, NumDims.x_Transpose_output_0);
Vars.x_conv_layers_0_Unsq = reshape(Vars.x_Transpose_output_0, shape);

% Set graph output arguments from Vars and NumDims:
x_conv_layers_0_Unsq = Vars.x_conv_layers_0_Unsq;
x_conv_layers_0_UnsqNumDims1024 = NumDims.x_conv_layers_0_Unsq;
% Set output state from Vars:
state = updateStruct(state, Vars);
end

function [inputDataPerms, outputDataPerms, Training] = parseInputs(input, numDataOutputs, params, varargin)
% Function to validate inputs to Transpose_To_UnsqueezeFcn:
p = inputParser;
isValidArrayInput = @(x)isnumeric(x) || isstring(x);
isValidONNXParameters = @(x)isa(x, 'ONNXParameters');
addRequired(p, 'input', isValidArrayInput);
addRequired(p, 'params', isValidONNXParameters);
addParameter(p, 'InputDataPermutation', 'auto');
addParameter(p, 'OutputDataPermutation', 'auto');
addParameter(p, 'Training', false);
parse(p, input, params, varargin{:});
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

function [input, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(input, params, varargin)
% Parse input arguments
[inputDataPerms, outputDataPerms, Training] = parseInputs(input, 1, params, varargin{:});
anyDlarrayInputs = any(cellfun(@(x)isa(x, 'dlarray'), {input}));
% Make the input variables into unlabelled dlarrays:
input = makeUnlabeledDlarray(input);
% Permute inputs if requested:
input = permuteInputVar(input, inputDataPerms{1}, 2);
end

function [x_conv_layers_0_Unsq] = postprocessOutput(x_conv_layers_0_Unsq, outputDataPerms, anyDlarrayInputs, Training, varargin)
% Set output type:
if ~anyDlarrayInputs && ~Training
    if isdlarray(x_conv_layers_0_Unsq)
        x_conv_layers_0_Unsq = extractdata(x_conv_layers_0_Unsq);
    end
end
% Permute outputs if requested:
x_conv_layers_0_Unsq = permuteOutputVar(x_conv_layers_0_Unsq, outputDataPerms{1}, 3);
end


%% dlarray functions implementing ONNX operators:

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

function [newShape, numDimsY] = prepareUnsqueezeArgs(X, ONNXAxes, numDimsX)
% Prepares arguments for implementing the ONNX Unsqueeze operator
numDimsY = numDimsX + numel(ONNXAxes);
ONNXAxes = extractdata(ONNXAxes);
ONNXAxes(ONNXAxes<0) = ONNXAxes(ONNXAxes<0) + numDimsY;
ONNXAxes = sort(ONNXAxes);                                              % increasing order
if numDimsY == 1
    newShape = size(X);
else
    DLTAxes  = flip(numDimsY - ONNXAxes);                                  % increasing order
    newShape = ones(1, numDimsY);
    posToSet = setdiff(1:numDimsY, DLTAxes, 'stable');
    newShape(posToSet) = size(X, 1:numel(posToSet));
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

function ussc001 = import_3D_point_matrix(filename, dataLines)
%IMPORTFILE Import data from a text file
%  USSC001 = IMPORTFILE(FILENAME) reads data from text file FILENAME for
%  the default selection.  Returns the numeric data.
%
%  USSC001 = IMPORTFILE(FILE, DATALINES) reads data for the specified
%  row interval(s) of text file FILENAME. Specify DATALINES as a
%  positive scalar integer or a N-by-2 array of positive scalar integers
%  for dis-contiguous row intervals.
%
%  Example:
%  ussc001 = import_3D_point_matrix("C:\Users\arosa\Box\USS Catheter\3d_data\data_11_08_2024_16_47_09\ussc_001.csv", [8, Inf]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 06-Dec-2024 10:47:59

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [8, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 20);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["FormatVersion", "VarName2", "TakeName", "ussc_004", "CaptureFrameRate", "VarName6", "ExportFrameRate", "VarName8", "CaptureStartTime", "Jan19701103370800", "TotalFramesInTake", "VarName12", "TotalExportedFrames", "VarName14", "RotationType", "XYZ", "LengthUnits", "Meters", "CoordinateSpace", "Global"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
ussc001 = readtable(filename, opts);

%% Convert to output type
ussc001 = table2array(ussc001);
end
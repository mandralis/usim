classdef Usim
    properties
        % inputs
        X;                % waveform data
        Y;                % curvature array
        wavelets;         % calibration wavelets in a cell array
        x_start;          % this might be unecessary
        
        % class properties
        N_samples;        % number of samples per waveform
        N_waveforms;      % number of waveforms
        N_wavelets;       % number of wavelets/surface features
        M;                % collocation matrix
        A;                % predicted amplitudes
        X_hat;            % predicted data based on fit
    end
    methods
        function A = forward(obj,X)
            A = pinv(obj.M)*X';
        end

        function X_hat = inverse(obj,A)
            X_hat = obj.M*A;
        end   

        function obj = Usim(wavelets,X,Y,x_start)
            % assign class properties
            obj.wavelets    = wavelets;
            obj.X           = X;
            obj.Y           = Y;
            obj.x_start     = x_start;
            obj.N_samples   = size(X,2);
            obj.N_waveforms = size(X,1);
            obj.N_wavelets  = length(wavelets);
            obj.M           = get_collocation_matrix(wavelets,obj.N_samples,x_start);
            
            % call forward method to get the wavelet amplitudes
            obj.A        = obj.forward(obj.X);

            % call predict method to see how well the original data was fit
            obj.X_hat    = obj.inverse(obj.A);
        end
    end
end
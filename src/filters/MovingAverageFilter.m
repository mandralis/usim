classdef MovingAverageFilter
    properties
        n_joints;  % dimension of state
        N;         % window size
        y;         % filter state
        xhistory;       % Nth previous input
    end
    methods
        function x_est = update(obj,x)
            obj.y = obj.y + 1/obj.N * (x - obj.xhistory(:,1));
            obj.xhistory(:,1) = [];            % delete first column from history
            obj.xhistory = [obj.xhistory, x];  % add new observation to history
            x_est = obj.y;
        end

        function obj = MovingAverageFilter(n_joints,N)
            % assign class properties
            obj.n_joints    = n_joints;
            obj.y           = zeros(n_joints,1);
            obj.N           = N;
            obj.xhistory    = zeros(n_joints,N);
        end
    end
end
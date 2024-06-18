classdef ExponentialSmoothingFilter
    properties
        n_joints;  % dimension of state
        s;         % filter state
        alpha;     % smoothing factor
    end
    methods
        function x_est = update(obj,x)
            obj.s = obj.s + obj.alpha * (x - obj.s);
            x_est = obj.s;
        end

        function obj = ExponentialSmoothingFilter(n_joints,alpha)
            % assign class properties
            obj.n_joints    = n_joints;
            obj.s           = zeros(n_joints,1);
            obj.alpha       = alpha;
        end
    end
end
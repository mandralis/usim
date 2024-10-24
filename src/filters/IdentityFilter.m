classdef IdentityFilter
    properties
    end
    methods
        function x_est = update(obj,zk)
            x_est = zk;
        end
    end
end
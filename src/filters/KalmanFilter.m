classdef KalmanFilter
    properties
        n_joints;
        F;
        H;
        Q;
        R;
        xkm1km1;
        Pkm1km1;
        xkkm1;
        Pkkm1;
    end
    methods
        function obj = predict(obj)
            obj.xkkm1 = obj.F*obj.xkm1km1;
            obj.Pkkm1 = obj.F*obj.Pkm1km1*obj.F' + obj.Q;
        end

        function obj = correct(obj,zk)
            yk_tilde = zk - obj.H * obj.xkkm1;
            Sk = obj.H * obj.Pkkm1 * obj.H' + obj.R;
            Kk = obj.Pkkm1 * obj.H' * inv(Sk);

            % updated estimate
            obj.xkm1km1 = obj.xkkm1 + Kk*yk_tilde;
            obj.Pkm1km1 = (eye(obj.n_joints) - Kk * obj.H) * obj.Pkkm1;
        end

        function x_est = update(obj,zk)
            obj = predict(obj);
            obj = correct(obj,zk);
            x_est = obj.xkm1km1;
        end

        function obj = KalmanFilter(n_joints,q,r)
            % assign class properties
            obj.n_joints    = n_joints;
            obj.F           = eye(n_joints);
            obj.H           = eye(n_joints);
            obj.xkm1km1     = zeros(n_joints,1);   % initial state estimate
            obj.Pkm1km1     = 0.5 * eye(n_joints); % initial covariance
            obj.xkkm1       = zeros(n_joints,1);   % will be overwritten
            obj.Pkkm1       = 0.5 * eye(n_joints); % will be overwritten
            obj.Q           = q*eye(n_joints);
            obj.R           = r*eye(n_joints);
        end
    end
end
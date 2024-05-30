function [T,N,B,k,t] = frenet(x,y,z)
    % FRENET - Frenet-Serret Spatial Curve Invariants
    %   
    %   [T,N,B,k,t] = frenet(x,y);
    %   [T,N,B,k,t] = frenet(x,y,z);
    % 
    %   Returns the 3 vector and 2 scalar invariants of a spatial curve defined
    %   by vectors x,y,z.  If z is omitted then the curve is only a 2D,
    %   but the equations are still valid.
    % 
    %    _    r'
    %    T = ----  (Tangent)
    %        |r'|
    % 
    %    _    T'
    %    N = ----  (Normal)
    %        |T'|
    %    _   _   _
    %    B = T x N (Binormal)
    % 
    %    k = |T'|  (Curvature)
    % 
    %    t = dot(-B',N) (Torsion)
    % 
    % 
    %    Example:
    %
    %    theta = 2*pi*linspace(0,2,100)';
    %    x = cos(theta);
    %    y = sin(theta);
    %    z = theta/(2*pi);
    %    [T,N,B,k,t] = frenet(x,y,z);
    %    line(x,y,z), hold on
    %    quiver3(x,y,z,T(:,1),T(:,2),T(:,3),'color','r')
    %    quiver3(x,y,z,N(:,1),N(:,2),N(:,3),'color','g')
    %    quiver3(x,y,z,B(:,1),B(:,2),B(:,3),'color','b')
    %    legend('Curve','Tangent','Normal','Binormal')
    % 
    % 
    
    if nargin == 2
        z = zeros(size(x));
    end
    
    % SPEED OF CURVE
    dx = gradient(x);
    dy = gradient(y);
    dz = gradient(z);
    dr = [dx dy dz];
    
    ddx = gradient(dx);
    ddy = gradient(dy);
    ddz = gradient(dz);
    ddr = [ddx ddy ddz];
    
    % tangent
    T = dr./mag(dr,3);
    
    % derivative of tangent
    dTx =  gradient(T(:,1));
    dTy =  gradient(T(:,2));
    dTz =  gradient(T(:,3));
    
    dT = [dTx dTy dTz];
    
    % normal
    N = dT./mag(dT,3);
    % binormal
    B = cross(T,N);
    % curvature
    % k = mag(dT,1);
    k = mag(cross(dr,ddr),1)./((mag(dr,1)).^3);
    % torsion
    t = dot(-B,N,2);

end

function N = mag(T,n)
    % vector norm (Nx3)
    %  M = mag(U)
    N = sum(abs(T).^2,2).^(1/2);
    d = find(N==0); 
    N(d) = eps*ones(size(d));
    N = N(:,ones(n,1));
end
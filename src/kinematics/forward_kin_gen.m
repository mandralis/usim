clear
syms a1 a2 a3 a4 a5 t1 t2 t3 t4

%initial conditions for each joint in the home position
theta  = [t1 t2 t3 t4];
q(:,1) = [0;a1;0];
q(:,2) = [0;a1+a2;0];
q(:,3) = [0;a1+a2+a3;0];
q(:,4) = [0;a1+a2+a3+a4;0];
q(:,5) = [0;a1+a2+a3+a4+a5;0];

%rotation axes of the joints in the home position 
w(:,1) = [0;0;1];
w(:,2) = [0;0;1];
w(:,3) = [0;0;1];
w(:,4) = [0;0;1];

%find initial twist matrices 
for i = 1:4
    v = cross(-w(:,i),q(:,i));
    g(:,:,i) = twist(w(:,i),v,theta(i));    
end

%gend(1:3,1:3) = [1 0 0;0 1 0;0 0 1];
gpre = [1 0 0 0;0 1 0 0;0 0 1 0; 0 0 0 1];
%pos(:,1) = [0;0;0];

%find the displacement of each joint by treating each joint as the final
%displacement of the tool frame
%also need to initialize the first position to be indicative of q1 pos

pos(:,1) = q(:,1);
for i = 1:4
    gend(1:3,4) = q(:,i+1);
    gend(4,1:4) = [0 0 0 1];
    gend(1:3,1:3) = [1 0 0;0 1 0;0 0 1];
    gpre = gpre* g(:,:,i); 
    gst(:,:,i) = gpre* gend;
    pos(:,i+1) = gst(1:3,4,i);
end

% make a matlab function for later use
forward_kinematics = matlabFunction(pos,"File","src/kinematics/forward_kin");

%function used to calculate twist matrices
function mat = twist(w,v,theta)
I = [1 0 0;0 1 0; 0 0 1];
what = [0 -w(3) w(2);w(3) 0 -w(1);-w(2) w(1) 0]; 
ewhat = I + what*sin(theta) + what^2*(1-cos(theta));
mat(1:3,1:3) = ewhat;
mat(4,1:4) = [0 0 0 1];
mat(1:3,4) = (I-ewhat)*cross(w,v)+ w*w.'*v*theta;
end



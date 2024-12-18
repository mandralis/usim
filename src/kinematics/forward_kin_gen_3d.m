clear all
close all
clc

% define number of joints of kinematic linkage
n_joints = 5;
n_per_joint = 2;
t = sym('t',[1,n_joints*n_per_joint]);
a = sym('a',[1,n_joints+1]);

%initial conditions for each joint in the home position
theta = t;
q_first     = sym(zeros(3,n_joints+1));
q_second  = sym(zeros(3,n_joints+1));

q_first(:,1) = [0;a(1);0];
q_second(:,1) = [0;a(1);0];

for i=1:n_joints
    q_first(:,i+1) = q_first(:,i) + [0;a(i+1);0];
    q_second(:,i+1) = q_first(:,i) + [0;a(i+1);0];
end

j =1 ;
for i = 1:2:n_joints*n_per_joint+1 

    q(:,i) = q_first(:,j);
    q(:,i+1) = q_second(:,j);
    j = j+1;
end
%%
%rotation axes of the joints in the home position 
w = zeros(3,n_joints*n_per_joint);
for i=1:2:n_joints*n_per_joint
    w(:,i) = [0;0;1];
    w(:,i+1) = [1;0;0];

end

%find initial twist matrices 
for i = 1:n_joints*n_per_joint
    v = cross(-w(:,i),q(:,i));
    g(:,:,i) = twist(w(:,i),v,theta(i));    
end

%gend(1:3,1:3) = [1 0 0;0 1 0;0 0 1];
% gpre = [1 0 0 0;0 1 0 0;0 0 1 0; 0 0 0 1];
gpre = eye(4);
%pos(:,1) = [0;0;0];

%find the displacement of each joint by treating each joint as the final
%displacement of the tool frame
%also need to initialize the first position to be indicative of q1 pos

pos(:,1) = q(:,1);
for i = 1:n_joints*n_per_joint
    gend(1:3,4) = q(:,i+1);
    gend(4,1:4) = [0 0 0 1];
    gend(1:3,1:3) = [1 0 0;0 1 0;0 0 1];
    gpre = gpre * g(:,:,i); 
    gst(:,:,i) = gpre * gend;
    pos(:,i+1) = gst(1:3,4,i);
end

% position for plot
pos_ = pos(:,1:2:end);

% make a matlab function for later use
forward_kinematics = matlabFunction(pos_,"File","src/kinematics/forward_kin_3d","Vars",{a,t});

%function used to calculate twist matrices
function mat = twist(w,v,theta)
    I = [1 0 0;0 1 0; 0 0 1];
    what = [0 -w(3) w(2);w(3) 0 -w(1);-w(2) w(1) 0]; 
    ewhat = I + what*sin(theta) + what^2*(1-cos(theta));
    mat(1:3,1:3) = ewhat;
    mat(4,1:4) = [0 0 0 1];
    mat(1:3,4) = (I-ewhat)*cross(w,v)+ w*w.'*v*theta;
end



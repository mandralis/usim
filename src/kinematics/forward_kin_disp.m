close all 
clear all
%set Robot arm parameters
d1 = 1;
a1 = 0.5;
a2 = 0.25;
a3 = 0.25;
%Designate desired point
x = 0.6;
y = 0.2;
z = 1.7;%z starts at 1

z = z-d1; 
ro = (x^2+y^2+z^2)^0.5; %check to see if the arm can reach the point normally 
if (a1+a2-a3)<=ro && ro<=(a1+a2+a3)
    fi = acos(z/ro);
    J = acos(((a1+a2)^2+ ro^2 - a3^2)/(2*(a1+a2)*ro));
    %If stamtems ensure that the arm travels to the correct quadrant
    if x>=0
        theta1 =  -pi/2 + asin(y/((x^2+y^2)^0.5));
    elseif x<0
        theta1 =  -(-pi/2 + asin(y/((x^2+y^2)^0.5)));
    end
    if z == ro
        theta1 = 0;
    end
    theta2 = -(J+(pi/2-fi));
    theta4 = pi - acos(((a1+a2)^2+ a3^2 - ro^2)/(2*(a1+a2)*a3));
    
    % theta1 = pi/4;
    % theta2 = -pi/4;
    % theta3 = pi/4;
    theta3 = 0;
     
    %this if statment encodes the second algorithm for reaching points
    %close to the first joint
    
elseif (a1+a2-a3)>=ro &&((a1-a2)-a3)<= ro 
    fi = acos(z/ro);
    J = acos(((a1-a2)^2+ ro^2 - a3^2)/(2*(a1-a2)*ro));
    if x>=0
        theta1 =  -pi/2 + asin(y/((x^2+y^2)^0.5));
    elseif x<0
        theta1 =  -(-pi/2 + asin(y/((x^2+y^2)^0.5)));
    end
    if z == ro
        theta1 = 0;
    end
    theta2 = -(J+(pi/2-fi));
    theta4 = acos(((a1-a2)^2+ a3^2 - ro^2)/(2*(a1-a2)*a3));
    theta3 = pi;
    
else %Detects whether no solution can be found 
    disp('No Solution can be Found')
    theta1 = 0;
    theta2 = 0;
    theta3 = 0;
    theta4 = 0;
end

theta1 = linspace(0,theta1);
theta2 = linspace(0,theta2);
theta3 = linspace(0,theta3);
theta4 = linspace(0,theta4);

%setting up th animation and the point marker
figure
hold on
pos = [0 0 0 0 0;0 0 0.5 0.75 1;0 1 1 1 1];
threed = plot3(pos(1,:),pos(2,:),pos(3,:));
axis([-1 1 -1 1 0 2 ]);
view([1.5,1.5,1.5])
threed.LineWidth = 10;
threed.Marker = 'o';
threed.MarkerEdgeColor = 'm';
dot = plot3(x,y,z+d1);
dot.Marker = 'o';
dot.MarkerFaceColor = 'r';
t2 = 0;
t3 = 0;
t4 = 0;

%the following for loops animate the robot arm moving starting from joint
%1 and ending with joint 4
for i = 1:length(theta1)
    t1 = theta1(i);
    
    xpos = [ 0,  0,                        -a1*cos(t2)*sin(t1), a1*cos(t1)*sin(t3) - (a1 + a2)*(cos(t1)*sin(t3) + cos(t2)*cos(t3)*sin(t1)) + a1*cos(t2)*sin(t1)*(cos(t3) - 1), (cos(t1)*sin(t3) + cos(t2)*cos(t3)*sin(t1))*(d1*sin(t4) + (a1 + a2)*(cos(t4) - 1)) - (cos(t4)*(cos(t1)*sin(t3) + cos(t2)*cos(t3)*sin(t1)) - sin(t1)*sin(t2)*sin(t4))*(a1 + a2 + a3) - d1*(sin(t4)*(cos(t1)*sin(t3) + cos(t2)*cos(t3)*sin(t1)) + cos(t4)*sin(t1)*sin(t2)) + a1*cos(t1)*sin(t3) + d1*sin(t1)*sin(t2) - sin(t1)*sin(t2)*(sin(t4)*(a1 + a2) - d1*(cos(t4) - 1)) + a1*cos(t2)*sin(t1)*(cos(t3) - 1)];
    ypos = [ 0,  0,                         a1*cos(t1)*cos(t2), a1*sin(t1)*sin(t3) - (a1 + a2)*(sin(t1)*sin(t3) - cos(t1)*cos(t2)*cos(t3)) - a1*cos(t1)*cos(t2)*(cos(t3) - 1), (sin(t1)*sin(t3) - cos(t1)*cos(t2)*cos(t3))*(d1*sin(t4) + (a1 + a2)*(cos(t4) - 1)) - d1*(sin(t4)*(sin(t1)*sin(t3) - cos(t1)*cos(t2)*cos(t3)) - cos(t1)*cos(t4)*sin(t2)) - (cos(t4)*(sin(t1)*sin(t3) - cos(t1)*cos(t2)*cos(t3)) + cos(t1)*sin(t2)*sin(t4))*(a1 + a2 + a3) - d1*cos(t1)*sin(t2) + a1*sin(t1)*sin(t3) + cos(t1)*sin(t2)*(sin(t4)*(a1 + a2) - d1*(cos(t4) - 1)) - a1*cos(t1)*cos(t2)*(cos(t3) - 1)];
    zpos = [ 0, d1, d1*cos(t2) - a1*sin(t2) - d1*(cos(t2) - 1),                          d1*cos(t2) - d1*(cos(t2) - 1) - cos(t3)*sin(t2)*(a1 + a2) + a1*sin(t2)*(cos(t3) - 1),                                                                                                                                            d1*(cos(t2)*cos(t4) - cos(t3)*sin(t2)*sin(t4)) - (cos(t2)*sin(t4) + cos(t3)*cos(t4)*sin(t2))*(a1 + a2 + a3) + cos(t2)*(sin(t4)*(a1 + a2) - d1*(cos(t4) - 1)) - d1*(cos(t2) - 1) + a1*sin(t2)*(cos(t3) - 1) + cos(t3)*sin(t2)*(d1*sin(t4) + (a1 + a2)*(cos(t4) - 1))];
    threed.XData = xpos;
    threed.YData = ypos;
    threed.ZData = zpos;
    drawnow limitrate
    pause(0.02)
end
for i = 1:length(theta2)
    t2 = theta2(i);
    
    xpos = [ 0,  0,                        -a1*cos(t2)*sin(t1), a1*cos(t1)*sin(t3) - (a1 + a2)*(cos(t1)*sin(t3) + cos(t2)*cos(t3)*sin(t1)) + a1*cos(t2)*sin(t1)*(cos(t3) - 1), (cos(t1)*sin(t3) + cos(t2)*cos(t3)*sin(t1))*(d1*sin(t4) + (a1 + a2)*(cos(t4) - 1)) - (cos(t4)*(cos(t1)*sin(t3) + cos(t2)*cos(t3)*sin(t1)) - sin(t1)*sin(t2)*sin(t4))*(a1 + a2 + a3) - d1*(sin(t4)*(cos(t1)*sin(t3) + cos(t2)*cos(t3)*sin(t1)) + cos(t4)*sin(t1)*sin(t2)) + a1*cos(t1)*sin(t3) + d1*sin(t1)*sin(t2) - sin(t1)*sin(t2)*(sin(t4)*(a1 + a2) - d1*(cos(t4) - 1)) + a1*cos(t2)*sin(t1)*(cos(t3) - 1)];
    ypos = [ 0,  0,                         a1*cos(t1)*cos(t2), a1*sin(t1)*sin(t3) - (a1 + a2)*(sin(t1)*sin(t3) - cos(t1)*cos(t2)*cos(t3)) - a1*cos(t1)*cos(t2)*(cos(t3) - 1), (sin(t1)*sin(t3) - cos(t1)*cos(t2)*cos(t3))*(d1*sin(t4) + (a1 + a2)*(cos(t4) - 1)) - d1*(sin(t4)*(sin(t1)*sin(t3) - cos(t1)*cos(t2)*cos(t3)) - cos(t1)*cos(t4)*sin(t2)) - (cos(t4)*(sin(t1)*sin(t3) - cos(t1)*cos(t2)*cos(t3)) + cos(t1)*sin(t2)*sin(t4))*(a1 + a2 + a3) - d1*cos(t1)*sin(t2) + a1*sin(t1)*sin(t3) + cos(t1)*sin(t2)*(sin(t4)*(a1 + a2) - d1*(cos(t4) - 1)) - a1*cos(t1)*cos(t2)*(cos(t3) - 1)];
    zpos = [ 0, d1, d1*cos(t2) - a1*sin(t2) - d1*(cos(t2) - 1),                          d1*cos(t2) - d1*(cos(t2) - 1) - cos(t3)*sin(t2)*(a1 + a2) + a1*sin(t2)*(cos(t3) - 1),                                                                                                                                            d1*(cos(t2)*cos(t4) - cos(t3)*sin(t2)*sin(t4)) - (cos(t2)*sin(t4) + cos(t3)*cos(t4)*sin(t2))*(a1 + a2 + a3) + cos(t2)*(sin(t4)*(a1 + a2) - d1*(cos(t4) - 1)) - d1*(cos(t2) - 1) + a1*sin(t2)*(cos(t3) - 1) + cos(t3)*sin(t2)*(d1*sin(t4) + (a1 + a2)*(cos(t4) - 1))];
    threed.XData = xpos;
    threed.YData = ypos;
    threed.ZData = zpos;
    drawnow limitrate
    pause(0.02)
end
for i = 1:length(theta3)
    t3 = theta3(i);
    
    xpos = [ 0,  0,                        -a1*cos(t2)*sin(t1), a1*cos(t1)*sin(t3) - (a1 + a2)*(cos(t1)*sin(t3) + cos(t2)*cos(t3)*sin(t1)) + a1*cos(t2)*sin(t1)*(cos(t3) - 1), (cos(t1)*sin(t3) + cos(t2)*cos(t3)*sin(t1))*(d1*sin(t4) + (a1 + a2)*(cos(t4) - 1)) - (cos(t4)*(cos(t1)*sin(t3) + cos(t2)*cos(t3)*sin(t1)) - sin(t1)*sin(t2)*sin(t4))*(a1 + a2 + a3) - d1*(sin(t4)*(cos(t1)*sin(t3) + cos(t2)*cos(t3)*sin(t1)) + cos(t4)*sin(t1)*sin(t2)) + a1*cos(t1)*sin(t3) + d1*sin(t1)*sin(t2) - sin(t1)*sin(t2)*(sin(t4)*(a1 + a2) - d1*(cos(t4) - 1)) + a1*cos(t2)*sin(t1)*(cos(t3) - 1)];
    ypos = [ 0,  0,                         a1*cos(t1)*cos(t2), a1*sin(t1)*sin(t3) - (a1 + a2)*(sin(t1)*sin(t3) - cos(t1)*cos(t2)*cos(t3)) - a1*cos(t1)*cos(t2)*(cos(t3) - 1), (sin(t1)*sin(t3) - cos(t1)*cos(t2)*cos(t3))*(d1*sin(t4) + (a1 + a2)*(cos(t4) - 1)) - d1*(sin(t4)*(sin(t1)*sin(t3) - cos(t1)*cos(t2)*cos(t3)) - cos(t1)*cos(t4)*sin(t2)) - (cos(t4)*(sin(t1)*sin(t3) - cos(t1)*cos(t2)*cos(t3)) + cos(t1)*sin(t2)*sin(t4))*(a1 + a2 + a3) - d1*cos(t1)*sin(t2) + a1*sin(t1)*sin(t3) + cos(t1)*sin(t2)*(sin(t4)*(a1 + a2) - d1*(cos(t4) - 1)) - a1*cos(t1)*cos(t2)*(cos(t3) - 1)];
    zpos = [ 0, d1, d1*cos(t2) - a1*sin(t2) - d1*(cos(t2) - 1),                          d1*cos(t2) - d1*(cos(t2) - 1) - cos(t3)*sin(t2)*(a1 + a2) + a1*sin(t2)*(cos(t3) - 1),                                                                                                                                            d1*(cos(t2)*cos(t4) - cos(t3)*sin(t2)*sin(t4)) - (cos(t2)*sin(t4) + cos(t3)*cos(t4)*sin(t2))*(a1 + a2 + a3) + cos(t2)*(sin(t4)*(a1 + a2) - d1*(cos(t4) - 1)) - d1*(cos(t2) - 1) + a1*sin(t2)*(cos(t3) - 1) + cos(t3)*sin(t2)*(d1*sin(t4) + (a1 + a2)*(cos(t4) - 1))];
    threed.XData = xpos;
    threed.YData = ypos;
    threed.ZData = zpos;
    drawnow limitrate
    pause(0.02)
end
for i = 1:length(theta4)
    t4 = theta4(i);
    
    xpos = [ 0,  0,                        -a1*cos(t2)*sin(t1), a1*cos(t1)*sin(t3) - (a1 + a2)*(cos(t1)*sin(t3) + cos(t2)*cos(t3)*sin(t1)) + a1*cos(t2)*sin(t1)*(cos(t3) - 1), (cos(t1)*sin(t3) + cos(t2)*cos(t3)*sin(t1))*(d1*sin(t4) + (a1 + a2)*(cos(t4) - 1)) - (cos(t4)*(cos(t1)*sin(t3) + cos(t2)*cos(t3)*sin(t1)) - sin(t1)*sin(t2)*sin(t4))*(a1 + a2 + a3) - d1*(sin(t4)*(cos(t1)*sin(t3) + cos(t2)*cos(t3)*sin(t1)) + cos(t4)*sin(t1)*sin(t2)) + a1*cos(t1)*sin(t3) + d1*sin(t1)*sin(t2) - sin(t1)*sin(t2)*(sin(t4)*(a1 + a2) - d1*(cos(t4) - 1)) + a1*cos(t2)*sin(t1)*(cos(t3) - 1)];
    ypos = [ 0,  0,                         a1*cos(t1)*cos(t2), a1*sin(t1)*sin(t3) - (a1 + a2)*(sin(t1)*sin(t3) - cos(t1)*cos(t2)*cos(t3)) - a1*cos(t1)*cos(t2)*(cos(t3) - 1), (sin(t1)*sin(t3) - cos(t1)*cos(t2)*cos(t3))*(d1*sin(t4) + (a1 + a2)*(cos(t4) - 1)) - d1*(sin(t4)*(sin(t1)*sin(t3) - cos(t1)*cos(t2)*cos(t3)) - cos(t1)*cos(t4)*sin(t2)) - (cos(t4)*(sin(t1)*sin(t3) - cos(t1)*cos(t2)*cos(t3)) + cos(t1)*sin(t2)*sin(t4))*(a1 + a2 + a3) - d1*cos(t1)*sin(t2) + a1*sin(t1)*sin(t3) + cos(t1)*sin(t2)*(sin(t4)*(a1 + a2) - d1*(cos(t4) - 1)) - a1*cos(t1)*cos(t2)*(cos(t3) - 1)];
    zpos = [ 0, d1, d1*cos(t2) - a1*sin(t2) - d1*(cos(t2) - 1),                          d1*cos(t2) - d1*(cos(t2) - 1) - cos(t3)*sin(t2)*(a1 + a2) + a1*sin(t2)*(cos(t3) - 1),                                                                                                                                            d1*(cos(t2)*cos(t4) - cos(t3)*sin(t2)*sin(t4)) - (cos(t2)*sin(t4) + cos(t3)*cos(t4)*sin(t2))*(a1 + a2 + a3) + cos(t2)*(sin(t4)*(a1 + a2) - d1*(cos(t4) - 1)) - d1*(cos(t2) - 1) + a1*sin(t2)*(cos(t3) - 1) + cos(t3)*sin(t2)*(d1*sin(t4) + (a1 + a2)*(cos(t4) - 1))];
    threed.XData = xpos;
    threed.YData = ypos;
    threed.ZData = zpos;
    drawnow limitrate
    pause(0.02)
end

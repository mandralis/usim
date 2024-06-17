close all 
clear all 

%Oscilloscope connect
%initialize Scope
myScope = oscilloscope();
availableResources = getResources(myScope);
myScope.Resource = availableResources{1};
connect(myScope);

%%
%Start viz of wire

figure
hold on
pos = [0 0 0 0 0 0;0 0.2 0.4 .6 .8 1;0 0 0 0 0 0];
threed = plot3(pos(1,:),pos(2,:),pos(3,:));
axis([-1 1 0 1.5 0 2 ]);
view([0,0,1.5])
threed.LineWidth = 3;
% threed.Marker = 'o';
% threed.MarkerEdgeColor = 'm';


while 1 == 1
    %%
    %Wavlet tranform of signal
    w = getWaveform(myScope, 'acquisition', true);
    
    spect = cwt(w,50000000);
    amp1 = abs(spect(63,4200));
    amp2 = abs(spect(63,6115));
    %compare change in amplitude
    
    
    %% update wire viz
    [xpos,ypos,zpos] = wirebend(-1.3*(amp1-0.0586),1.3*(amp2-0.2114),0.2);
    
    threed.XData = xpos;
    threed.YData = ypos;
    threed.ZData = zpos;
    drawnow limitrate
    pause(0.05)
    
end

function [xpos,ypos,zpos] = wirebend(theta1,theta2,L)
t1 = theta1;
t2 = theta1;

t3 = theta2;
t4 = theta2;

a1 = L;
a2 = L;
a3 = L;
a4 = L;
a5 = L;

%Padded with extra zeros at begginging for initial post
xpos = [0, 0,       a1*sin(t1) - sin(t1)*(a1 + a2),       a1*sin(t1) - (cos(t1)*sin(t2) + cos(t2)*sin(t1))*(a1 + a2 + a3) + cos(t1)*sin(t2)*(a1 + a2) + sin(t1)*(a1 + a2)*(cos(t2) - 1),       a1*sin(t1) - (cos(t3)*(cos(t1)*sin(t2) + cos(t2)*sin(t1)) + sin(t3)*(cos(t1)*cos(t2) - sin(t1)*sin(t2)))*(a1 + a2 + a3 + a4) + cos(t1)*sin(t2)*(a1 + a2) + sin(t3)*(cos(t1)*cos(t2) - sin(t1)*sin(t2))*(a1 + a2 + a3) + sin(t1)*(a1 + a2)*(cos(t2) - 1) + (cos(t3) - 1)*(cos(t1)*sin(t2) + cos(t2)*sin(t1))*(a1 + a2 + a3),       a1*sin(t1) - (cos(t4)*(cos(t3)*(cos(t1)*sin(t2) + cos(t2)*sin(t1)) + sin(t3)*(cos(t1)*cos(t2) - sin(t1)*sin(t2))) + sin(t4)*(cos(t3)*(cos(t1)*cos(t2) - sin(t1)*sin(t2)) - sin(t3)*(cos(t1)*sin(t2) + cos(t2)*sin(t1))))*(a1 + a2 + a3 + a4 + a5) + sin(t4)*(cos(t3)*(cos(t1)*cos(t2) - sin(t1)*sin(t2)) - sin(t3)*(cos(t1)*sin(t2) + cos(t2)*sin(t1)))*(a1 + a2 + a3 + a4) + cos(t1)*sin(t2)*(a1 + a2) + sin(t3)*(cos(t1)*cos(t2) - sin(t1)*sin(t2))*(a1 + a2 + a3) + (cos(t3)*(cos(t1)*sin(t2) + cos(t2)*sin(t1)) + sin(t3)*(cos(t1)*cos(t2) - sin(t1)*sin(t2)))*(cos(t4) - 1)*(a1 + a2 + a3 + a4) + sin(t1)*(a1 + a2)*(cos(t2) - 1) + (cos(t3) - 1)*(cos(t1)*sin(t2) + cos(t2)*sin(t1))*(a1 + a2 + a3)];
ypos = [0,a1, cos(t1)*(a1 + a2) - a1*(cos(t1) - 1), (cos(t1)*cos(t2) - sin(t1)*sin(t2))*(a1 + a2 + a3) - a1*(cos(t1) - 1) + sin(t1)*sin(t2)*(a1 + a2) - cos(t1)*(a1 + a2)*(cos(t2) - 1), (cos(t3)*(cos(t1)*cos(t2) - sin(t1)*sin(t2)) - sin(t3)*(cos(t1)*sin(t2) + cos(t2)*sin(t1)))*(a1 + a2 + a3 + a4) - a1*(cos(t1) - 1) + sin(t1)*sin(t2)*(a1 + a2) + sin(t3)*(cos(t1)*sin(t2) + cos(t2)*sin(t1))*(a1 + a2 + a3) - cos(t1)*(a1 + a2)*(cos(t2) - 1) - (cos(t3) - 1)*(cos(t1)*cos(t2) - sin(t1)*sin(t2))*(a1 + a2 + a3), (cos(t4)*(cos(t3)*(cos(t1)*cos(t2) - sin(t1)*sin(t2)) - sin(t3)*(cos(t1)*sin(t2) + cos(t2)*sin(t1))) - sin(t4)*(cos(t3)*(cos(t1)*sin(t2) + cos(t2)*sin(t1)) + sin(t3)*(cos(t1)*cos(t2) - sin(t1)*sin(t2))))*(a1 + a2 + a3 + a4 + a5) - a1*(cos(t1) - 1) + sin(t4)*(cos(t3)*(cos(t1)*sin(t2) + cos(t2)*sin(t1)) + sin(t3)*(cos(t1)*cos(t2) - sin(t1)*sin(t2)))*(a1 + a2 + a3 + a4) + sin(t1)*sin(t2)*(a1 + a2) + sin(t3)*(cos(t1)*sin(t2) + cos(t2)*sin(t1))*(a1 + a2 + a3) - (cos(t3)*(cos(t1)*cos(t2) - sin(t1)*sin(t2)) - sin(t3)*(cos(t1)*sin(t2) + cos(t2)*sin(t1)))*(cos(t4) - 1)*(a1 + a2 + a3 + a4) - cos(t1)*(a1 + a2)*(cos(t2) - 1) - (cos(t3) - 1)*(cos(t1)*cos(t2) - sin(t1)*sin(t2))*(a1 + a2 + a3)];
zpos = [0, 0,                                    0,                                                                                                                                   0,                                                                                                                                                                                                                                                                                                                                0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               0];

end
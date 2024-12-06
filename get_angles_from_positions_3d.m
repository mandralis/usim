clear all
close all
clc

    % get angle 
    Theta_zx(i,:) = -atan(diff(z)./diff(x));
    Theta_yx(i,:) = -atan(diff(y)./diff(x));

    % convert to relative angle for kinematic linkage
    Theta_relative(i,:) = [Theta(i,1), diff(Theta(i,:))];

   
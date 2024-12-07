function [Theta_relative_zx,Theta_relative_yx] = get_angles_from_positions_3d(Position_matrix,n_valid_triggers_per_acquisition_cycle)

%measure the angels inbetween the 3d points to find join angles for the
%kinematic linkage

Theta_zx = zeros(n_valid_triggers_per_acquisition_cycle,5);
Theta_yx = zeros(n_valid_triggers_per_acquisition_cycle,5);

Theta_relative_zx = zeros(n_valid_triggers_per_acquisition_cycle,5);
Theta_relative_yx = zeros(n_valid_triggers_per_acquisition_cycle,5);
for i = 1:n_valid_triggers_per_acquisition_cycle
    x = Position_matrix(1,:,i);
    y = Position_matrix(2,:,i);
    z = Position_matrix(3,:,i);
    % get angle
    Theta_zx(i,:) = -atan(diff(z)./diff(x));
    Theta_yx(i,:) = -atan(diff(y)./diff(x));
    
    % convert to relative angle for kinematic linkage
    Theta_relative_zx(i,:) = [Theta_zx(i,1), diff(Theta_zx(i,:))];
    Theta_relative_yx(i,:) = [Theta_yx(i,1), diff(Theta_yx(i,:))];
    
end

end
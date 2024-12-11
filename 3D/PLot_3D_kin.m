clear
close all
clc

data_path = 'data_11_08_2024_16_47_09\';
fname = [get_local_data_path(),data_path];
addpath(genpath(fname));

load([fname,'Position_matrix_tot.mat']);
load([fname,'Theta_relative_yx.mat']);
load([fname,'Theta_relative_zx.mat']);


%%
a(1:6) = 0.1;

point_num = 1;

for i = 1:10:length(Theta_relative_yx)
    
    subplot(2,1,1)
    scatter(Position_matrix_tot(1,:,i)-Position_matrix_tot(1,point_num,i),Position_matrix_tot(2,:,i)-Position_matrix_tot(2,point_num,i))
    axis([-0.3,0.7,-.3,0.3])
    hold on
    Pkin = forward_kin5(a,Theta_relative_yx(i,:));
    plot(Pkin(2,:)-Pkin(2,1),Pkin(1,:)-Pkin(1,1),'Marker','o','MarkerFaceColor','k',"Color",'r','MarkerEdgeColor','k',LineWidth=1.0);
    
    
    
    subplot(2,1,2)
    scatter(Position_matrix_tot(1,:,i)-Position_matrix_tot(1,point_num,i),Position_matrix_tot(3,:,i)-Position_matrix_tot(3,point_num,i))
    axis([-0.3,0.7,-.3,0.3])
    hold on
    Pkin = forward_kin5(a,Theta_relative_zx(i,:));
    plot(Pkin(2,:)-Pkin(2,1),Pkin(1,:)-Pkin(1,1),'Marker','o','MarkerFaceColor','k',"Color",'r','MarkerEdgeColor','k',LineWidth=1.0);
    
    drawnow
    pause(0.01)
    clf
end
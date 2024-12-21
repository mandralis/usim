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
close all

a(1:6) = [0,0.11,0.097,0.11,0.11,0.11];

point_num = 1;

figure 

for i =1:10:length(Theta_relative_yx)
%     
%     subplot(2,2,1)
%     scatter(Position_matrix_tot(1,:,i)-Position_matrix_tot(1,point_num,i),Position_matrix_tot(2,:,i)-Position_matrix_tot(2,point_num,i))
%     axis([-0.3,0.7,-.3,0.3])
%     hold on
%     Pkin = forward_kin5(a,Theta_relative_yx(i,:));
%     plot(Pkin(2,:)-Pkin(2,1),Pkin(1,:)-Pkin(1,1),'Marker','o','MarkerFaceColor','k',"Color",'r','MarkerEdgeColor','k',LineWidth=1.0);
%     
%     
%      
%     subplot(2,2,2)
%     scatter(Position_matrix_tot(1,:,i)-Position_matrix_tot(1,point_num,i),Position_matrix_tot(3,:,i)-Position_matrix_tot(3,point_num,i))
%     axis([-0.3,0.7,-.3,0.3])
%     hold on
%     Pkin = forward_kin5(a,Theta_relative_zx(i,:));
%     plot(Pkin(2,:)-Pkin(2,1),Pkin(1,:)-Pkin(1,1),'Marker','o','MarkerFaceColor','k',"Color",'r','MarkerEdgeColor','k',LineWidth=1.0);
%     
    k = 1;
    for j = 1:2:9
        theta(j) = Theta_relative_zx(i,k);
        theta(j+1) = Theta_relative_yx(i,k);
        k = k+1;
    
    end
    
%     subplot(2,2,3)
    b = Position_matrix_tot(:,:,i);
    scatter3(b(1,:)- b(1,1),b(2,:)-b(2,1),b(3,:)-b(3,1))
    axis([0 .6 -.6 .6 -.6 .6]);
    hold on
    Pkin = forward_kin_3d(a,theta);
    plot3(Pkin(2,:)-Pkin(2,1),-(Pkin(3,:)-Pkin(3,1)),Pkin(1,:)-Pkin(1,1),'Marker','o','MarkerFaceColor','k',"Color",'r','MarkerEdgeColor','k',LineWidth=1.0);
%     axis([0 0.6 -1 1 -1 1 ]);
    view([1,1,1])
    
    drawnow
    pause(0.01)
    clf
    
end
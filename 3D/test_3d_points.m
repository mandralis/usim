
%%
a = table2array(ussc001);


for i = 1:442
    
    b = reshape(a(i,3:20),3,6);
    scatter3(b(1,:),b(2,:),b(3,:))
    axis([-.4 .4 -.4 .4 -.4 .4]);
    pause(0.01)
end

figure
for i = 3000:5:19950
    plot(smooth(X(i,1:1300)))
    axis([1 1300 -1.5 1.5])
    i
    pause(0.1)
    
end
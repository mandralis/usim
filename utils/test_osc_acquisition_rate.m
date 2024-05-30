clear all 
close all
clc

myScope = connect();

N = 3;
dt_vec = [];
for i=1:N
    disp(i)
    tstart = tic;
    readWaveform(myScope, 'acquisition', true);
    tend = toc(tstart);
    dt_vec = [dt_vec tend];
end
disp(1/mean(dt_vec))
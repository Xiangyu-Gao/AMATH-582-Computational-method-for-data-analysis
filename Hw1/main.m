% Clean workspace
clc
clear all; 
close all; 

% Imports the data as the 262144x49 (space by time) matrix called subdata
load 'C:\Users\Xiangyu Gao\Downloads\subdata\subdata.mat'; 

L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); x = x2(1:n); y =x; z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

for j=1:49
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    M = max(abs(Un),[],'all');
    close all
    isosurface(X,Y,Z,abs(Un)/M,0.7)
    axis([-10 10 -10 10 -10 10])
    grid on
    drawnow
    pause(1)
end
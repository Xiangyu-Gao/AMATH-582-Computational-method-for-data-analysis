% Clean workspace
clc
clear
close all 

% Imports the data as the 262144x49 (space by time) matrix called subdata
load 'C:\Users\Xiangyu Gao\Downloads\subdata\subdata.mat'; 

%% Problem 1
L = 10; % spatial domain
n = 64; % Fourier modes

% reshape subdata
Un = zeros(n, n, n, 49);
for j=1:49
    Un(:,:,:,j)=reshape(subdata(:,j),n,n,n);
end

% % Visualization
% x2 = linspace(-L,L,n+1); x = x2(1:n); y =x; z = x;
% [X,Y,Z]=meshgrid(x,y,z);
% for j=1:49
%     M = max(abs(Un(:, :, :, j)),[],'all');
%     close all
%     isosurface(X,Y,Z,abs(squeeze(abs(Un(:, :, :, j))))/M,0.7);
%     axis([-10 10 -10 10 -10 10])
%     grid on
%     drawnow
%     pause(1)
% end

% FFT on reshaped data
Un_fft = fftshift(fft(Un, 64, 4));
Un_fft_avg = squeeze(sum(abs(Un_fft), [1 2 3]));
Un_fft_avg = Un_fft_avg / max(Un_fft_avg);

% Plot averaged spectrum
ks = (2*pi/49)*[0:(n/2-1) -n/2:-1];
ks = fftshift(ks);
figure(1)
plot(ks, Un_fft_avg)
axis([-4, 4, 0.99, 1])
xlabel('wavenumber (k)')
ylabel('|ut|/max(|ut|)')
title('averaging spectrum')

% Find the center frequency generated by submarine
[~, I] = max(Un_fft_avg)
ks(I)

%% Problem 2
% Gaussian filter
filter = exp(-0.3*(ks-ks(I)).^2);
filter = reshape(filter, 1, 1, 1, n);
Un_fft_filt = filter .* Un_fft;

% plot filtered data
Un_fft_filt_avg = squeeze(sum(abs(Un_fft_filt), [1 2 3]));
Un_fft_filt_avg = Un_fft_filt_avg / max(Un_fft_filt_avg);

figure(2)
plot(ks, Un_fft_filt_avg)
axis([-4, 4, 0.99, 1])
xlabel('wavenumber (k)')
ylabel('|ut|/max(|ut|)')
title('filtered averaging spectrum')

% ifftshift, ifft
Un_filt = ifft(ifftshift(Un_fft_filt), 49, 4);

% determine the path of submarine
P_vec = [];
for i = 1:49
    [~, P] = max(abs(squeeze(Un_filt(:,:,:,i))), [], 'all', 'linear');
    P_vec = [P_vec, P];
end

% tranform liner indices to subscripts
sz = [n, n, n];
[I1, I2, I3] = ind2sub(sz, P_vec);

% % Visualization
% x2 = linspace(-L,L,n+1); x = x2(1:n); y =x; z = x;
% [X,Y,Z]=meshgrid(x,y,z);
% for j=1:49
%     M = max(abs(Un_filt(:, :, :, j)),[],'all');
%     close all
%     isosurface(X,Y,Z,abs(squeeze(abs(Un_filt(:, :, :, j))))/M,0.7);
%     axis([-10 10 -10 10 -10 10])
%     grid on
%     drawnow
%     pause(1)
% end

% plot the path of the submarine
figure(3)
plot3(I1, I2, I3, '-.')
xlabel('x-index'), ylabel('y-index'), zlabel('z-index')
axis([35, 55, 9, 45, 5, 55])
title('path of the submarine')

%% Problem 3
% Get the numerical x-, y-coordinates to track the submarine
x2 = linspace(-L,L,n+1); x = x2(1:n); y =x; z = x;
figure(4)
plot(x(I1), x(I2), '-o')
xlabel('x spatial domain'), ylabel('y spatial domain')
title('x-y path of the sub-tracking aricraft')

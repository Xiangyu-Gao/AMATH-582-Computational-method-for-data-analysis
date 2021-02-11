clc
clear all
close all

%% Read input data
[y, Fs] = audioread('C:\Users\Xiangyu Gao\Desktop\AMATH\AMATH 582\hw2\Floyd.m4a');
tr_gnr = length(y)/Fs; % record time in seconds
figure(1)
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Comfortably Numb');
% p8 = audioplayer(y,Fs); playblocking(p8);

%% Parameter CONSTRUCTION
y = y(1:length(y)-1);
n = length(y);
L = n/Fs; % second
t2 = linspace(0,L,n+1); 
t = t2(1:n); 
k = (2*pi/L)*[0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

%% SLIDING GABOR WINDOW
Sgt_spec = []; 
tslide = 0:0.5:L;

for j = 1:length(tslide)
    g = exp(-3000*(t-tslide(j)).^2); % Gabor 
    Sg = g .* y'; 
    Sgt = fft(Sg); 
    Sgt_spec = [Sgt_spec; log(abs(fftshift(Sgt))+1)]; 
end

% figure(2)
% pcolor(tslide, ks, Sgt_spec.'), 
% shading interp 
% set(gca,'Ylim',[0, 3e4],'Fontsize',[14]) 
% colormap(hot)
% colorbar
% caxis([1.0 2.5])
% xlabel('time (s)')
% ylabel('wavenumber (k)')
% title('Music score of Floyd')

%% Isolate Bass
filtered_range = [0.3e4, 0.6e4];
filter = zeros(size(ks));
[~, id1] = min(abs(ks-filtered_range(1)));
[~, id2] = min(abs(ks-filtered_range(2)));
filter(id1:id2) = 1;

figure(3)
Sgt_spec_filter = Sgt_spec .* filter;
pcolor(tslide, ks, Sgt_spec_filter.'), 
shading interp 
set(gca,'Ylim',[0, 3e4],'Fontsize',[14]) 
colormap(hot)
colorbar
caxis([1.0 2.5])
xlabel('time (s)')
ylabel('wavenumber (k)')
title('Music score of Floyd with filtering out overtones')

%% Obtain Guitar
filtered_range = [2e4, 3e4];
filter = zeros(size(ks));
[~, id1] = min(abs(ks-filtered_range(1)));
[~, id2] = min(abs(ks-filtered_range(2)));
filter(id1:id2) = 1;

figure(4)
Sgt_spec_filter = Sgt_spec .* filter;
pcolor(tslide(1:30), ks, Sgt_spec_filter(1:30,:).')
shading interp 
set(gca,'Ylim',[0, 3e4],'Fontsize',[14]) 
colormap(hot)
colorbar
caxis([0.5 2.5])
xlabel('time (s)')
ylabel('wavenumber (k)')
title('Music score of Guitar solo in Floyd clip')

figure(5)
Sgt_spec_filter = Sgt_spec .* filter;
pcolor(tslide(31:60), ks, Sgt_spec_filter(31:60,:).')
shading interp 
set(gca,'Ylim',[0, 3e4],'Fontsize',[14]) 
colormap(hot)
colorbar
caxis([0.5 2.5])
xlabel('time (s)')
ylabel('wavenumber (k)')
title('Music score of Guitar solo in Floyd clip')

figure(6)
Sgt_spec_filter = Sgt_spec .* filter;
pcolor(tslide(61:90), ks, Sgt_spec_filter(61:90,:).')
shading interp 
set(gca,'Ylim',[0, 3e4],'Fontsize',[14]) 
colormap(hot)
colorbar
caxis([0.5 2.5])
xlabel('time (s)')
ylabel('wavenumber (k)')
title('Music score of Guitar solo in Floyd clip')

figure(7)
Sgt_spec_filter = Sgt_spec .* filter;
pcolor(tslide(91:120), ks, Sgt_spec_filter(91:120,:).')
shading interp 
set(gca,'Ylim',[0, 3e4],'Fontsize',[14]) 
colormap(hot)
colorbar
caxis([0.5 2.5])
xlabel('time (s)')
ylabel('wavenumber (k)')
title('Music score of Guitar solo in Floyd clip')

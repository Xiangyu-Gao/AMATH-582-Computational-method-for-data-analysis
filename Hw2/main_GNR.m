clc
clear all
close all

%% Read input data
[y, Fs] = audioread('C:\Users\Xiangyu Gao\Desktop\AMATH\AMATH 582\hw2\GNR.m4a');
tr_gnr = length(y)/Fs; % record time in seconds
figure(1)
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Sweet Child O Mine');
% p8 = audioplayer(y,Fs); playblocking(p8);

%% Parameter CONSTRUCTION
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

figure(2)
pcolor(tslide, ks, Sgt_spec.'), 
shading interp 
set(gca,'Ylim',[0, 5e4],'Fontsize',[14]) 
colormap(hot)
colorbar
caxis([1.0 2.5])
xlabel('time (s)')
ylabel('wavenumber (k)')
title('Music score of GNR')

%% filter out overtones and Isolate Guitar
filtered_range = [1.5e4, 3e4];
filter = zeros(size(ks));
[~, id1] = min(abs(ks-filtered_range(1)));
[~, id2] = min(abs(ks-filtered_range(2)));
filter(id1:id2) = 1;

figure(3)
Sgt_spec_filter = Sgt_spec .* filter;
pcolor(tslide, ks, Sgt_spec_filter.'), 
shading interp 
set(gca,'Ylim',[0, 5e4],'Fontsize',[14]) 
colormap(hot)
colorbar
caxis([1.0 2.5])
xlabel('time (s)')
ylabel('wavenumber (k)')
title('Music score of GNR with filtering out overtones')




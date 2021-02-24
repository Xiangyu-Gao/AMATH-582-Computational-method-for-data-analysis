clc
clear all
close all

data = load('C:\Users\Xiangyu Gao\Downloads\582-movie\cam3_4.mat');
vidFrames = data.vidFrames3_4;
numFrames = size(vidFrames, 4);

% cam2_1
% y_range = [110, 370];
% x_range = [260, 350];

% cam3_1
% y_range = [250, 320];
% x_range = [280, 480];

% cam1_2
% y_range = [230, 350];
% x_range = [310, 450];

% cam2_2
% y_range = [130, 350];
% x_range = [200, 400];

% cam3_2
% y_range = [130, 350];
% x_range = [300, 450];

% cam1_3
% y_range = [230, 400];
% x_range = [260, 400];

% cam2_3
% y_range = [180, 400];
% x_range = [200, 400];

% cam3_3
% y_range = [130, 350];
% x_range = [300, 450];

% cam1_4
% y_range = [230, 400];
% x_range = [310, 450];

% cam2_4
% y_range = [130, 350];
% x_range = [200, 400];

% cam3_4
y_range = [130, 350];
x_range = [300, 510];

for k = 1 : numFrames
    mov(k).cdata = vidFrames(:,:,:,k);
    mov(k).colormap = [];
end

x_inds_all = [];
y_inds_all = [];

for j = 1:numFrames
    X = frame2im(mov(j)); 
    J = rgb2gray(X);
    J(1:y_range(1), :) = 0;
    J(y_range(2):end, :) = 0;
    J(:, 1:x_range(1)) = 0;
    J(:, x_range(2):end) = 0;
%     imshow(J); drawnow
    % find the location of white spot top on the mass
    inds = find(J>=235);
    if length(inds) < 1
%         imshow(J); drawnow
        max(J, [], 'all')
        continue
    end
    [y_ind, x_ind] = ind2sub(size(J), inds);
    x_inds_all = [x_inds_all, mean(x_ind)];
    y_inds_all = [y_inds_all, mean(y_ind)];
end

figure(1)
plot(y_inds_all)
save('cam3_4_x.mat', 'x_inds_all', '-v6')
save('cam3_4_y.mat', 'y_inds_all', '-v6')
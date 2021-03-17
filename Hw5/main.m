clc
clear all
close all

%% read movie file
% a = VideoReader('monte_carlo_low.mp4');
a = VideoReader('ski_drop_low.mp4');

X = [];
for img = 1:a.NumFrames
    filename=strcat('frame',num2str(img),'.jpg');
    b = rgb2gray(read(a, img));
    % reshape each frame and concatenate them
    X = [X, reshape(b, [540*960, 1])];
    
end
X = single(X);

dt = 1;
t = [1:size(X, 2)];
%% DMD
X1 = X(:,1:end-1); X2 = X(:,2:end);
[U2,Sigma2,V2] = svd(X1, 'econ');

r = 3; 
U = U2(:,1:r); 
Sigma = Sigma2(1:r,1:r); 
V=V2(:,1:r);

Atilde = U'*X2*V/Sigma;
[W,D] = eig(Atilde);
Phi = X2*V/Sigma*W;

mu = diag(D);
omega = log(mu)/dt;

figure(1)
plot(abs(omega), 'o')
title('absolute value of \omega')
xlabel('index')
ylabel('value')


u0 = X(:, 1);
y0 = Phi\u0;  % pseudo-inverse initial conditions

u_modes = zeros(r,length(t));  % DMD reconstruction for every time point
for iter = 1:length(t)
    u_modes(:,iter) = y0.*exp(omega*t(iter));
end
u_dmd = Phi(:, 1)*u_modes(1, :);   % DMD resconstruction with all modes

%% Make real-valued reconstruction
X_lowrank = abs(u_dmd);
X_sparse = X - X_lowrank;

% Set all positives of sparse to zero
X_sparse_new = min(X_sparse, zeros(size(X_sparse)));
R = X_sparse - X_sparse_new;
X_lowrank_new = X_lowrank + R;
X_reconstruct = X_lowrank_new + X_sparse_new;

%% show the foreground and background\
% show movies
% for i = 1:length(t)
%     image = abs(X_sparse_new(:, i));
%     imshow(uint8(reshape(image, [540, 960])))
% end

show_image_index = 100;

% foreground
figure(2)
image = abs(X_sparse_new(:, show_image_index));
imshow(uint8(reshape(image, [540, 960])))

% background
figure(3)
image = abs(X_lowrank_new(:, show_image_index));
imshow(uint8(reshape(image, [540, 960])))

% reconstrcution image
figure(4)
image = abs(X_reconstruct(:, show_image_index));
imshow(uint8(reshape(image, [540, 960])))
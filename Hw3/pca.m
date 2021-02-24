clc
clear all
close all

%% read the saved data
x_inds_all_1 = load('cam1_4_x.mat').x_inds_all;
x_inds_all_2 = load('cam2_4_x.mat').x_inds_all;
x_inds_all_3 = load('cam3_4_x.mat').x_inds_all;

y_inds_all_1 = load('cam1_4_y.mat').y_inds_all;
y_inds_all_2 = load('cam2_4_y.mat').y_inds_all;
y_inds_all_3 = load('cam3_4_y.mat').y_inds_all;

len = min([length(x_inds_all_1), length(x_inds_all_2), length(x_inds_all_3)]);

X = [x_inds_all_1(1:len); y_inds_all_1(1:len); x_inds_all_2(1:len);  ...
    y_inds_all_2(1:len); x_inds_all_3(1:len); y_inds_all_3(1:len)];

%% Eigenvalue Decompsition
[m, n] = size(X);
mn = mean(X, 2); % compute mean for each row
X = X - repmat(mn, 1, n); % subtract mean

Cx = (1/(n-1)) * X * X'; % covariance
[V,D] = eig(Cx); % eigenvectors(V)/eigenvalues(D)
lambda = diag(D); % get eigenvalues

%% PCA analysis
[dummy, m_arrange] = sort(-1*lambda); % sort in decreasing order
lambda = lambda(m_arrange);
V = V(:, m_arrange);
Y = V' * X; % produce the principal components projection

figure(1)
plot(Y(1, :),'LineWidth',1)
hold on
plot(Y(2, :),'LineWidth',1)
plot(Y(3, :),'LineWidth',1)
plot(Y(4, :),'LineWidth',1)

xlabel('time')
ylabel('position')
title('Principal motion components of horizontal displacement and rotation case')
% title('Principal motion components of noisy case')
legend('Mode 1', 'Mode 2', 'Mode 3', 'Mode 4')

figure(2)
scatter(m_arrange, lambda, 'filled')
xlabel('index of principal components')
ylabel('corresponding eigenvalues')
set(gca,'xtick',1:6)
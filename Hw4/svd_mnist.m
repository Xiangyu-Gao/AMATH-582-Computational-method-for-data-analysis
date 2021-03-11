clear all
close all
clc

%% Problem 1
% Load images
[images_train, labels_train] = mnist_parse('mnist\train-images.idx3-ubyte', 'mnist\train-labels.idx1-ubyte');
[images_test, labels_test] = mnist_parse('mnist\t10k-images.idx3-ubyte', 'mnist\t10k-labels.idx1-ubyte');

images = cat(3, images_train(:,:,1:50000), images_test);
labels = cat(1, labels_train(1:50000,:), labels_test);

% Size of each picture
m = 28;
n = 28;
% reshape images
A = double(reshape(images, [m*n, length(labels)]));

%% Problem 2
% remove the averaged image (row-wise)
mean_data = mean(A,2);
A = A - mean_data;

% Computing the SVD
[U,S,V] = svd(A, 0);

% PCA analysis and plot singular value spectrum
sigval_spct = diag(S).^2;
ALL_energy = sum(sigval_spct);
sum(sigval_spct(1:26)) / ALL_energy % Rank 26, %70 variance

figure(1)
plot(sigval_spct(1:50), 'o')
set(gca, 'YScale', 'log')
xlabel('Index of singular value')
ylabel('Singular value')
title('The singular values spectrum')

% Reconstruct images with PCA basis
num_PCAbasis = 26;
image_id = 2;
image1 = U(:, 1:num_PCAbasis)*S(1:num_PCAbasis, 1:num_PCAbasis)*V(image_id, 1:num_PCAbasis)';
figure(2)
subplot(1,2,1)
imshow(uint8(reshape(image1,m,n)))
subplot(1,2,2)
imshow(uint8(reshape(A(:, image_id),m,n)))

%% Problem 3
% interpretate the U, S, V matrix
numImg = size(A, 2);
Phi = U;
Phi(:,1) = -1*Phi(:,1);
% plot the first 9 reshaped coumns of matrix U
figure(3)
count = 1;
for i=1:3
    for j=1:3
        subplot(3,3,count)
        imshow(uint8(25000*reshape(Phi(:,count),m,n)));
        count = count + 1;
    end
end

%% Problem 4
% Projection onto 3 V-modes
figure(4)
for label=0:9
    label_indices = find(labels == label);
    plot3(V(label_indices, 2), V(label_indices, 3), V(label_indices, 5),...
        'o', 'DisplayName', sprintf('%i',label), 'Linewidth', 0.5)
    hold on
end
xlabel('2nd V-Mode'), ylabel('3rd V-Mode'), zlabel('5th V-Mode')
title('Projection onto V-modes 2, 3, 5')
legend
set(gca,'Fontsize', 10)

%% Save projection onto V-modes for classification training, testing
V_save = V(:, 1:num_PCAbasis);
save('V_mode.mat', 'V_save')
save('labels.mat', 'labels')




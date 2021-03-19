%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Regression and Sparsity for Digit Classification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all

% read training data
[images_train, labels_train] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
[images_test, labels_test] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');

% Size of each picture
m = 28;
n = 28;
% reshape images
X = double(reshape(images_train, [m*n, length(labels_train)]));

% remove the averaged image (row-wise)
mean_data = mean(X, 2);
X = X - mean_data;

B_label = zeros(10, length(labels_train));% matrix B
for i = 1:length(labels_train)
    if labels_train(i) > 0
        B_label(labels_train(i), i) = 1;
    else
        B_label(10, i) = 1;
    end
end

% Solve AX = B
A = B_label * pinv(X);

%% Test
% reshape  test images
X_test = double(reshape(images_test, [m*n, length(labels_test)]));

% remove the averaged image (row-wise)
X_test = X_test - mean_data;

% test
B_test = A * X_test;

% predict
test_soft = softmax(B_test);
[~, testI] = max(test_soft, [], 1);
testI = mod(testI, 10);

% calculate accuracy for all 10 digits
YPred = testI';
YValidation = labels_test;
accuracy = sum(YPred == YValidation)/numel(YValidation);

%% accuracy for individual digits
accuracy_ind=zeros(1,10);
for i=0:9
    idx = find(labels_test == i);
    YPred_digit = testI(idx)';
    YValidation_digit = labels_test(idx);
    accuracy_ind(i+1) = sum(YPred_digit == YValidation_digit)/numel(YValidation_digit);
end

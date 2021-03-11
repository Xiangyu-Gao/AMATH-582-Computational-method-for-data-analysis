clc
clear all
close all

% read saved data
load('V_mode.mat');
load('labels.mat');
V_train = V_save(1:50000, :);
V_test = V_save(50001:end, :);
labels_train = labels(1:50000, :);
labels_test = labels(50001:end, :);

%% Linear Discriminate classifier (LDA) for classification of two digits
digit1 = 0;
digit2 = 1;

digit1_train_idx = find(labels_train == digit1);
digit2_train_idx = find(labels_train == digit2);
digit1_test_idx = find(labels_test == digit1);
digit2_test_idx = find(labels_test == digit2);

x_test = [V_test(digit1_test_idx, :); V_test(digit2_test_idx, :)];
x_train = [V_train(digit1_train_idx, :); V_train(digit2_train_idx, :)];
ctest = [labels_test(digit1_test_idx, :); labels_test(digit2_test_idx, :)];
ctrain = [labels_train(digit1_train_idx, :); labels_train(digit2_train_idx, :)];

% LDA classifier
pre = classify([x_test; x_train], x_train, ctrain);
% calculate accuracy for testing and training
accu_test = sum(pre(1:length(x_test)) == ctest)/length(x_test);
accu_train = sum(pre(length(x_test)+1:end) == ctrain)/length(x_train);

%% Linear Discriminate classifier (LDA) for classification of thress digits
digit1 = 0;
digit2 = 1;
digit3 = 2;

digit1_train_idx = find(labels_train == digit1);
digit2_train_idx = find(labels_train == digit2);
digit3_train_idx = find(labels_train == digit3);
digit1_test_idx = find(labels_test == digit1);
digit2_test_idx = find(labels_test == digit2);
digit3_test_idx = find(labels_test == digit3);

x_test = [V_test(digit1_test_idx, :); V_test(digit2_test_idx, :); ...
    V_test(digit3_test_idx, :)];
x_train = [V_train(digit1_train_idx, :); V_train(digit2_train_idx, :); ...
    V_train(digit3_train_idx, :)];
ctest = [labels_test(digit1_test_idx, :); labels_test(digit2_test_idx, :); ...
    labels_test(digit3_test_idx, :)];
ctrain = [labels_train(digit1_train_idx, :); labels_train(digit2_train_idx, :); ...
    labels_train(digit3_train_idx, :)];

% LDA classifier
pre = classify([x_test; x_train], x_train, ctrain);
% calculate accuracy for testing and training
accu_test = sum(pre(1:length(x_test)) == ctest)/length(x_test);
accu_train = sum(pre(length(x_test)+1:end) == ctrain)/length(x_train);

%% find the hardest and easiest pair of two digits with LDA
train_accu_list = zeros(10, 10);
test_accu_list = zeros(10, 10);

for i=1:10
    for j=1:10
        if j <= i
            continue
        end
        digit1 = i-1;
        digit2 = j-1;

        digit1_train_idx = find(labels_train == digit1);
        digit2_train_idx = find(labels_train == digit2);
        digit1_test_idx = find(labels_test == digit1);
        digit2_test_idx = find(labels_test == digit2);

        x_test = [V_test(digit1_test_idx, :); V_test(digit2_test_idx, :)];
        x_train = [V_train(digit1_train_idx, :); V_train(digit2_train_idx, :)];
        ctest = [labels_test(digit1_test_idx, :); labels_test(digit2_test_idx, :)];
        ctrain = [labels_train(digit1_train_idx, :); labels_train(digit2_train_idx, :)];

        % LDA classifier
        pre = classify([x_test; x_train], x_train, ctrain);
        % calculate accuracy for testing and training
        accu_test = sum(pre(1:length(x_test)) == ctest)/length(x_test);
        accu_train = sum(pre(length(x_test)+1:end) == ctrain)/length(x_train);
        train_accu_list(i, j) = accu_train;
        test_accu_list(i, j) = accu_test;
    end
end


%% SVM classifier
% SVM classifier with training data, labels and test set
x_train = V_train;
ctrain = labels_train;
x_test = V_test;
ctest = labels_test;

Mdl = fitcecoc(x_train, ctrain);
test_pre = predict(Mdl, x_test);
train_pre = predict(Mdl, x_train);

accu_test = sum(test_pre == ctest)/length(x_test);
accu_train = sum(train_pre == ctrain)/length(x_train);

%% Decision tree classifier
% classification tree on fisheriris data
tree = fitctree(x_train, ctrain);
% view(tree,'mode','graph');

test_pre = predict(tree, x_test);
train_pre = predict(tree, x_train);

accu_test = sum(test_pre == ctest)/length(x_test);
accu_train = sum(train_pre == ctrain)/length(x_train);

%% SVM on the hardest and easiest pair of digits to separate
% digit1 = 4; % hardest
% digit2 = 9;
digit1 = 0; % easiest
digit2 = 1;

digit1_train_idx = find(labels_train == digit1);
digit2_train_idx = find(labels_train == digit2);
digit1_test_idx = find(labels_test == digit1);
digit2_test_idx = find(labels_test == digit2);

x_test = [V_test(digit1_test_idx, :); V_test(digit2_test_idx, :)];
x_train = [V_train(digit1_train_idx, :); V_train(digit2_train_idx, :)];
ctest = [labels_test(digit1_test_idx, :); labels_test(digit2_test_idx, :)];
ctrain = [labels_train(digit1_train_idx, :); labels_train(digit2_train_idx, :)];

Mdl = fitcsvm(x_train, ctrain);
test_pre = predict(Mdl, x_test);
train_pre = predict(Mdl, x_train);

accu_test = sum(test_pre == ctest)/length(x_test);
accu_train = sum(train_pre == ctrain)/length(x_train);

%% Decision tree
tree = fitctree(x_train, ctrain);
% view(tree,'mode','graph');

test_pre = predict(tree, x_test);
train_pre = predict(tree, x_train);

accu_test = sum(test_pre == ctest)/length(x_test);
accu_train = sum(train_pre == ctrain)/length(x_train);
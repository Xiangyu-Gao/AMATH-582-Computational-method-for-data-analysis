close all; clear; clc
%% import data
[images_train, labels_train] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
[images_test, labels_test] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');

% reshape images
images_train = double(reshape(images_train, [28*28, length(labels_train)]));
images_test = double(reshape(images_test, [28*28, length(labels_test)]));

% create categorical label
cat_label_train = categorical(labels_train);

%% FCN
% define the network
layers = [
    featureInputLayer(28*28, 'Name','input')
    fullyConnectedLayer(1024, 'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(512, 'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(128, 'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(10, 'Name','fc4')
    softmaxLayer('Name','sm')
    classificationLayer('Name','classification')]

% training options
options = trainingOptions('adam', ...
    'InitialLearnRate',0.005, ...
    'MaxEpochs',1, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% train the network
net = trainNetwork(images_train',cat_label_train,layers,options);

% calculate accuracy
YPred = single(classify(net, images_test'));
YPred = YPred - 1;
YValidation = labels_test;
accuracy = sum(YPred == YValidation)/numel(YValidation);

%% accuracy for individual digits
accuracy_ind=zeros(1,10);
for i=0:9
    idx = find(labels_test == i);
    YPred_digit = YPred(idx);
    YValidation_digit = labels_test(idx);
    accuracy_ind(i+1) = sum(YPred_digit == YValidation_digit)/numel(YValidation_digit);
end

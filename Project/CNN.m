close all; clear; clc
%% import data
[images_train, labels_train] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
[images_test, labels_test] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
    
% tranform MNIST matlab data to image files
if exist('MNIST_images','dir')==0
    for i=0:9
         label = num2str(i);
         mkdir(fullfile(pwd,'MNIST_images', 'train',label));
         mkdir(fullfile(pwd,'MNIST_images', 'test',label));
    end
    num=0;
    for i=1:length(labels_train)
        num=num+1;
        label=num2str(labels_train(i));
        name=fullfile(pwd,'MNIST_images','train',label,['images_' label '_' num2str(num) '.png']);
        imwrite(images_train(:,:,i), name);
    end
    num=0;
    for i=1:length(labels_test)
        num=num+1;
        label=num2str(labels_test(i));
        name=fullfile(pwd,'MNIST_images','test',label,['images_' label '_' num2str(num) '.png']);
        imwrite(images_test(:,:,i), name);
    end
else
end

%% CNN
% create data store
train_db=imageDatastore(fullfile(pwd,'MNIST_images','train'), 'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames');
test_db=imageDatastore(fullfile(pwd,'MNIST_images','test'), 'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames');

% define the network
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

% training options
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',1, ...
    'Shuffle','every-epoch', ...
    'ValidationData',test_db, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% train the network
net = trainNetwork(train_db,layers,options);

% calculate accuracy
YPred = classify(net,test_db);
YValidation = test_db.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);

%% CNN on individual digits
accuracy_ind=zeros(1,10);
for i=0:9
    test_db=imageDatastore(fullfile(pwd,'MNIST_images','test', num2str(i)), 'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames');
    YPred = classify(net,test_db);
    YValidation = test_db.Labels;

    accuracy(i+1) = sum(YPred == YValidation)/numel(YValidation);
end

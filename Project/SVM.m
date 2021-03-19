close all; clear; clc
%% import data
[images_train, labels_train] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
[images_test, labels_test] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');


%% %% part I
[a_train,b_train,c_train]=size(images_train);
[a_test,b_test,c_test]=size(images_test);
data_train = zeros(a_train*b_train, c_train);
data_test = zeros(a_test*b_test, c_test);
for i=1:1:c_train
    data_train(:,i)=reshape(images_train(:,:,i),a_train*b_train, 1);
end
for i=1:1:c_test
    data_test(:,i)=reshape(images_test(:,:,i),a_test*b_test, 1);
end
% SVD
[U,S,V]=svd(double(data_train),'econ');
proj=(S*V')';
% singular value spectrum
plot(diag(S)/sum(diag(S)), '-o')
xlabel('Singular value','Fontsize',12)
ylabel('Proportion','Fontsize',12)

%% compare different digit pairs
accuracy_svm=zeros(10,10);
for i=0:1:8
    for j=i+1:1:9
        x1_train=proj(labels_train==i,2:10);
        x2_train=proj(labels_train==j,2:10);
        [len1,temp]=size(x1_train);
        [len2,temp]=size(x2_train);
        xtrain=[x1_train; x2_train];
        ctrain=[i*ones(len1,1); j*ones(len2,1)];

        xtest_temp=(U'*data_test)';
        x1_test=xtest_temp(labels_test==i,2:10);
        x2_test=xtest_temp(labels_test==j,2:10);
        [len1,temp]=size(x1_test);
        [len2,temp]=size(x2_test);
        xtest=[x1_test; x2_test];
        ctest=[i*ones(len1,1); j*ones(len2,1)];

        Mdl_svm = fitcsvm(xtrain,ctrain,'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);

        pre=predict(Mdl_svm,xtest);

        errorNum=sum(abs(ctest-pre)>0);
        accuracy_svm(i+1,j+1)=1-errorNum/length(ctest);
    end
end

%% SVM, all digits
xtrain=proj(:,2:10)/max(max(S));
xtest_temp=(U'*data_test)';
xtest=xtest_temp(:,2:10)/max(max(S));

SVMModels = cell(10,1);
classes = 0:1:9;
rng(1); % For reproducibility
for j = 1:numel(classes)
    indx = labels_train==classes(j); % Create binary classes for each classifier
    SVMModels{j} = fitcsvm(xtrain,indx,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);
end

for j = 1:numel(classes)
    [~,score] = predict(SVMModels{j},xtest);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end

[~,maxScore] = max(Scores,[],2);
errorNum=sum(abs(labels_test+1-maxScore)>0);
accuracy_svm=1-errorNum/length(labels_test)

%% SVM, individual accuracy
accuracy_ind=zeros(1,10);
for i=0:9
    xtest=xtest_temp(labels_test==i,2:10)/max(max(S));
    [len1,temp]=size(xtest);
    ctest=i*ones(len1,1);
    Scores=[];
    for j = 1:numel(classes)
        [~,score] = predict(SVMModels{j},xtest);
        Scores(:,j) = score(:,2);
    end
    [~,maxScore] = max(Scores,[],2);
    errorNum=sum(abs(ctest+1-maxScore)>0);
    accuracy_ind(i+1)=1-errorNum/length(ctest);
end
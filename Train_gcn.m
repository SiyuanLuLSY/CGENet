function [YPred_GCN, Accuracy_GCN, TrainedNet] = Train_gcn(TrainedNet, options, Trainset, Testset, batch_size, num_neighbors,epoches)
for m =1:epoches
    rowrank = randperm(size(Trainset.feature, 1)); % 随机打乱的数字，从1~行数打乱
    Train_features = Trainset.feature(rowrank, :);
    Train_labels = Trainset.labels(rowrank);
    num_batches = ceil(size(Trainset.feature,1)/batch_size);
 for i=1:num_batches-1
%      close all;
    temp_features = Train_features((i-1)*batch_size+1:i*batch_size,:);
    temp_Label = Train_labels((i-1)*batch_size+1:i*batch_size,1);
    
    L = Laplacian(temp_features,num_neighbors);
%     L = Laplacian(temp_features,C,num_neighbors);
    temp_input = L*temp_features;  
    Input = zeros(1,1,size(temp_features,2),size(temp_features,1));
    for j =1:size(temp_features,1)
        Input(1,1,:,j) = temp_input(j,:)';
    end
    if ~(i==1 && m==1)
        TrainedNet = layerGraph(TrainedNet);
    end
%      TrainedNet = trainNetwork(Input,temp_Label,TrainedNet,options);
    TrainedNet = trainNetwork(Input,temp_Label,TrainedNet,options);
%     YPred = classify(TrainedNet, augimdsValidation);
%     accuracy = mean(YPred == imdsValidation.Labels);
 end
end

num_batches = ceil(size(Testset.feature,1)/batch_size);

YPred_GCN = [];
 for i=1:num_batches-1
    temp_features = Testset.feature((i-1)*batch_size+1:i*batch_size,:);
    L = Laplacian(temp_features,num_neighbors);
%     temp_Label = Test_labels((i-1)*num_clusters+1:i*num_clusters,1);
    temp_input = L*temp_features;  
    
    Test_input =  zeros(1,1,size(temp_features,2),size(temp_features,1));    
    for j =1:size(temp_input,1)
        Test_input(1,1,:,j) = temp_input(j,:)';
    end
    temp_pred = classify(TrainedNet, Test_input);
    YPred_GCN = cat(1,YPred_GCN,temp_pred);
 end
    temp_features = Testset.feature(end-batch_size+1:end,:);
    L = Laplacian(temp_features,num_neighbors);
%     temp_Label = Testset.labels(end-num_clusters+1:end,1);
    temp_input = L*temp_features;  
    
    Test_input =  zeros(1,1,size(temp_features,2),size(temp_features,1));    
    for j =1:size(temp_input,1)
        Test_input(1,1,:,j) = temp_input(j,:)';
    end
    
    temp_pred = classify(TrainedNet, Test_input);
    length = size(Testset.labels,1) - (num_batches-1)*batch_size;
    YPred_GCN = cat(1,YPred_GCN,temp_pred(end-length+1:end,1));
    Accuracy_GCN = mean(YPred_GCN == Testset.labels)
end
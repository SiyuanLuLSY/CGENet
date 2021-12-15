function [cfmelm, cfmsnn, cfmrvfl, cfmen] = GCNetwork(activation_layer,TrainedNet,nnet,imdsTrain, imdsValidation,options, batch_size, num_neighbors,epoches)
% activation_layer = 'fc128';
inputSize = TrainedNet.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,'ColorPreprocessing','gray2rgb');
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation,'ColorPreprocessing','gray2rgb');
layer = activation_layer;
Trainset.labels = imdsTrain.Labels;
Testset.labels = imdsValidation.Labels;
Trainset.feature = activations(TrainedNet,augimdsTrain,layer,'OutputAs','rows');
Testset.feature = activations(TrainedNet, augimdsValidation,layer, 'OutputAs','rows');
% epoches = 40 ;
% batch_size = 32;
% num_neighbors = 8;
N=512;
GCNet = layerGraph(nnet);
[ cfmelm, cfmsnn, cfmrvfl, cfmen] = Train_gcn_elm(N,Trainset, Testset, num_neighbors)
%[YPred_GCN, Accuracy_GCN,TrainGCNet] = Train_gcn(N, GCNet, options, Trainset, Testset, batch_size, num_neighbors,epoches);
end
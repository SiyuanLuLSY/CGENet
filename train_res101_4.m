clear all;
% load eff00;
% load eff;
load res18;
% load res18-4;
%  load mobilenet;
%  load alex;
% load dense201;
% load res50;
% load vgg;
% load cam;
% load ziquanzhu;
% lgraph = lgraph_1;

tic
 load nnet128;
imds = imageDatastore('.\p','IncludeSubfolders',true,'LabelSource','foldernames');
[imd1 imd2 imd3 imd4 imd5] = splitEachLabel(imds,0.2,0.2,0.2,0.2,0.2,'randomize');
partStores{1} = imd1.Files ;
partStores{2} = imd2.Files ;
partStores{3} = imd3.Files ;
partStores{4} = imd4.Files ;
partStores{5} = imd5.Files ;

load seed;
rng(s);
numClasses = numel(categories(imds.Labels));
inputSize = lgraph.Layers(1).InputSize;


k = 5;
idx = crossvalind('Kfold', k, k);

batch_size = 32;
num_neighbors = 3;
epoches = 4;
activation_layer = 'fc256';

options2 = trainingOptions('adam', ...
'MiniBatchSize',batch_size, ...
'MaxEpochs',epoches, ...
'InitialLearnRate',1e-4, ...
'Shuffle','every-epoch', ...
 'LearnRateSchedule','piecewise', ...
'LearnRateDropFactor',0.5, ...
'LearnRateDropPeriod',5, ...   
'Verbose',false);

    for i = 1:k
        i
      test_idx = (idx == i);
      train_idx = ~test_idx;
      imdsTest = imageDatastore(partStores{test_idx}, 'IncludeSubfolders', true,'LabelSource', 'foldernames');
      imdsTrain = imageDatastore(cat(1, partStores{train_idx}), 'IncludeSubfolders', true,'LabelSource', 'foldernames');
      augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,'ColorPreprocessing','gray2rgb');
      augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsTest,'ColorPreprocessing','gray2rgb');
       options1 = trainingOptions('adam', ...
    'MiniBatchSize',24, ...
    'MaxEpochs',1, ...
        'ValidationData',augimdsValidation,...    
    'ValidationFrequency',400,...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...    
    'Plots','training-progress',...
    'Verbose',false);
TrainedNet_4{i} = trainNetwork(augimdsTrain,lgraph,options1);
           [YPred,scores] = classify(TrainedNet_4{i},augimdsValidation);  
      cfmres(:,:,i) = confusionmat(imdsTest.Labels, YPred)
[cfmelm(:,:,i),cfmsnn(:,:,i),cfmrvfl(:,:,i),cfmen(:,:,i)] = GCNetwork(activation_layer,...
          TrainedNet_4{i},nnet128,imdsTrain, imdsTest,options2, batch_size,...
          num_neighbors,epoches);
             [sensitivity_net(i),specificity_net(i),accuracy_net(i),...
           precision_net(i),F1_net(i)]=getindexes(cfmelm(:,:,i));
%       Test_labels_4{i} = imdsTest.Labels;
%       [sen1_4{i}, spe1_4{i}, accuracy1_4{i}, precision1_4{i}, f1_4{i}]=cal_sen_spe_pre__acc_F1(Test_labels_4{i},YPred2_4{i});
    end
m_acc=mean(accuracy_net)
m_sen=mean(sensitivity_net)
m_spe=mean(specificity_net)
m_pre=mean(precision_net)
m_F1=mean(F1_net)
toc
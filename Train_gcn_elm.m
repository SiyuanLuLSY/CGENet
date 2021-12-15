function [ cfmelm, cfmsnn, cfmrvfl, cfmen] = Train_gcn_elm(N,Trainset, Testset, num_neighbors)
    Train_features = Trainset.feature;
    Train_labels = single(Trainset.labels);
    L_train = Laplacian(Train_features,num_neighbors);
    Train_input=L_train * Train_features;

    Test_features = Testset.feature;
    Test_labels = single(Testset.labels);
    L_Test = Laplacian(Test_features,num_neighbors);
    Test_input=L_Test * Test_features;
    M1=[ Train_labels Train_input];
    M2=[ Test_labels Test_input ];
[TY1,FP1,FN1,TrainingTime1, TestingTime1, TrainingAccuracy1, TestingAccuracy1] = elm(M1,...
    M2, 1, N, 'sig');

[TY2,FP2,FN2,TrainingTime2, TestingTime2, TrainingAccuracy2, TestingAccuracy2] = snn(M1,...
    M2, 1, N, 'sig');

option.ActivationFunction='sigmoid';option.N=N;
[train_accuracy3,TestingAccuracy3,TY3]=RVFL_train_val(Train_features,...
    Train_labels,Test_features,Test_labels,option);
% res=sum(abs(TY-TTest));

TY1=TY1';
[b1,ind1]=max(TY1,[],2);
cfmelm = confusionmat(Test_labels, single(ind1));
TY2=TY2';
[b2,ind2]=max(TY2,[],2);
cfmsnn = confusionmat(Test_labels, single(ind2));
ind3=TY3;
cfmrvfl = confusionmat(Test_labels, single(ind3));
ind=[ind1 ind2 ind3];
ta=0;
for i=1:size(ind,1)
table = tabulate(ind(i,:));
[maxCount,idx] = max(table(:,2));
y=table(idx);
if maxCount==1 y=1;end
ta=[ta;y];
end
ta=ta(2:end);
ta=ta';
toc
cfmen = confusionmat(Test_labels, single(ta))

end
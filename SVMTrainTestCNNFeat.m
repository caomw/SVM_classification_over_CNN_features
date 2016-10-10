% -------------------------------------------------------------------------
% This file takes in the train and test CNN features and the corresponding
% groundtruth labels, and trains and tests with Pegasos SVM Method.
% 
% NOTE - Instead of CNN features, we can input any types of features, but
% we use it normally with CNN features.
%
% Plots the SVM performance for all classes and labels. 
% Also plots average feature vectors for classes under a one-vs-all
% scenario, that is used to train SVMs. 
% ---------------------------
% INPUTS : 
% (a) XTrain as NTrain x d matrix, N = number of train examples, d = dim of each example
% (b) XTest as NTest x d matrix, N = number of test examples, d = dim of each example 
% (c) YTrain as NTrain x 1 matrix - Groundtruth for training examples
% (d) YTest as NTest x 1 matrix - Groundtruth for test examples
% (e) tags as M x 1 Cell Array - Names of all labels, used for display
% (f) outputFolder = Folder path for saving the outputs
% (g) styleString = String name used during saving 
% 
% M = number of labels
% 
% NOTE - For ground truth N x 1 matrix, then contents should have 
% label numbering starting from 0 to M - 1
% ---------------------------
% OUTPUTS : 
% (a) For each class / label, trained [W,b] are saved. 
% (b) Plots showing accuracy for each class / label are saved 
% (c) A Mat file saving the class / label wise accuracy and total accuracy
% (d) Plots showing average features across classes are saved 
% ---------------------------
% Author : Sukrit Shankar 
% -------------------------------------------------------------------------
function SVMTrainTestCNNFeat (XTrain, XTest, YTrain, YTest, ...
    tags, outputFolder, styleString)  

% --------------------------------------------------------
% Configuration Settings
numberOfLabels = length(tags); 
maxNumberOfTrainingImages = 12000;  % For efficiency of Pegasos
outputFolderName_a = 'trainedModels'; 
outputFolderName_b = 'perClassAccuracyPlots'; 
outputFolderName_c = 'condensedOutputs'; 
outputFolderName_d = 'classWiseOneVsAllFeatCompare'; 

% --------------------------------------------------------
% Make appropriate output directories 
mkdir (strcat(outputFolder,'/',styleString,'_SVMs')); 
mkdir (strcat(outputFolder,'/',styleString,'_SVMs/',outputFolderName_a)); 
mkdir (strcat(outputFolder,'/',styleString,'_SVMs/',outputFolderName_b));
mkdir (strcat(outputFolder,'/',styleString,'_SVMs/',outputFolderName_c)); 
mkdir (strcat(outputFolder,'/',styleString,'_SVMs/',outputFolderName_d)); 

% --------------------------------------------------------
% Train and test with SVMs
posCases = 0; totalCases = 0; 
labelWiseAccuracy = zeros(1,numberOfLabels); 
for m = 1:1:numberOfLabels
    fprintf('\n Doing the SVM Training for Label = %d',m); 
    tic
    
    % -------------------------
    % Train SVM for the label m 
    temp =  find(YTrain == m-1);
    indicesPos = temp(1:min(length(temp),maxNumberOfTrainingImages/2)); 
    clear temp; 
    temp =  find(YTrain ~= m-1);
    indicesNeg = temp(1:min(length(temp),maxNumberOfTrainingImages/2)); 
    clear temp; 
    
    for i = 1:1:length(indicesPos)
        trainFeat(i,:) = XTrain(indicesPos(i),:); 
        trainLabels(i,:) = 1; 
    end
    for i = 1:1:length(indicesNeg)
        trainFeat(length(indicesPos)+i,:) = XTrain(indicesNeg(i),:); 
        trainLabels(length(indicesPos)+i,:) = -1; 
    end
    
    % Calculate the mean features
    trainFeatMeanPos(m,:) = mean (trainFeat(1:length(indicesPos),:),1); 
    trainFeatMeanNeg(m,:) = mean (trainFeat(length(indicesPos)+1:end,:),1); 
    clear indicesPos indicesNeg; 
    
    % Call SVM module 
    [W,b] = pegasosSVMTrain(trainFeat,trainLabels);
    
    % Save 
    save (strcat(outputFolder,'/',styleString,'_SVMs/',outputFolderName_a,...
        '/modelForLabel_',num2str(m),'.mat'),'W','b'); 
    
    % -------------------------
    % Test SVM for the label m 
    % We need to test only positive cases 
    temp =  find(YTest == m-1);
    indicesPos = temp(1:end); 
    clear temp;  
    
    for i = 1:1:length(indicesPos)
        testLabels(i,:) = 1; 
    end
    for i = 1:1:length(indicesPos)
        q = XTest(indicesPos(i),:); 
        testFeat(i,:) = q(1,:);    
        clear q; 
    end
    
    % Calculate the mean features
    testFeatMeanPos(m,:) = mean (testFeat(1:end,:),1); 
    clear indicesPos indicesNeg; 

    % Infer over trained SVM module 
    YPred = SVMTest(testFeat,W,b);  
    
    % Compute the accuracy for the attribute 
    diff = YPred - testLabels; 
    labelWiseAccuracy(m) = numel(find(diff == 0)) / size(testLabels,1); 
    
    % Add the number of positive cases and the number of total cases
    posCases = posCases + numel(find(diff == 0)); 
    totalCases = totalCases + size(testLabels,1); 
    
    % Clear variables for the loop 
    clear W b YPred trainFeat testFeat testLabels trainLabels;
    toc
end  

% Compute total accuracy 
totalAccuracy = posCases / totalCases; 

% Save the condensed outputs 
save (strcat(outputFolder,'/',styleString,'_SVMs/',outputFolderName_c,...
    '/SVMBinaryClassificationResults.mat'),'labelWiseAccuracy','totalAccuracy'); 

% --------------------------------------------------------------
% Save output plots for Per Class Accuracy 
rootFolderName = strcat(outputFolder,'/',styleString,'_SVMs/',outputFolderName_d); 
for m = 1:1:numberOfLabels
    figure; 
    subplot (3,1,1); bar (testFeatMeanPos(m,:)); 
    title ('Testing - Mean Positive Label Feature');
    
    subplot (3,1,2); bar (trainFeatMeanPos(m,:)); 
    title ('TRAINING - Mean Positive Label Feature');  
    
    subplot (3,1,3); bar (trainFeatMeanNeg(m,:)); 
    title ('TRAINING - Mean Negative Label Feature'); 
    
    set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 30 20])
    print('-dpng', strcat(rootFolderName,'/featuresForLabel_',...
        num2str(m), '.png'), '-r300');
    clf; close all; 
end
clear rootFolderName; 

% --------------------------------------------------------------
% Save output plots for Mean Feature Comparison 
rootFolderName = strcat(outputFolder,'/',styleString,'_SVMs/',outputFolderName_b); 
plotLabelAccuracy(tags,styleString,labelWiseAccuracy,totalAccuracy,rootFolderName); 
clear rootFolderName; 













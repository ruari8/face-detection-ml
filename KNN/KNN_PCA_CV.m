%% Clear workspace and close all figures
clear all;
close all;
clc;
rng(40);

%% Load all data
% Load the complete dataset
[allImages, allLabels] = loadFaceImages2('face_train_6040.cdataset', 1);
[testImages, testLabels] = loadFaceImages2('face_test_6040.cdataset', 1);

% Apply PCA to the complete dataset
fprintf('Applying PCA to the dataset...\n');

% Standardize the dataset
allMean = mean(allImages, 1);
allStd = std(allImages, 0, 1);
standardizedImages = (allImages - allMean) ./ allStd;

% Compute PCA
[coeff, score, ~, ~, explained] = pca(standardizedImages);

% Determine the number of components to retain 95% variance
cumulativeExplained = cumsum(explained);
numComponents = find(cumulativeExplained >= 95, 1);

% Reduce dimensionality of the dataset
reducedImages = score(:, 1:numComponents);

% Number of folds for cross-validation
k = 10;

% Initialize arrays to store metrics for each fold
accuracies = zeros(k, 1);
precisions = zeros(k, 1);
recalls = zeros(k, 1);
specificities = zeros(k, 1);
f1Scores = zeros(k, 1);
aucs = zeros(k, 1);

% Create cross-validation partition
cvp = cvpartition(allLabels, 'KFold', k);

%% Perform k-fold cross-validation
for fold = 1:k
    fprintf('\nProcessing fold %d/%d...\n', fold, k);
    
    % Get training and validation indices for this fold
    trainIdx = training(cvp, fold);
    validIdx = test(cvp, fold);
    
    % Split data for this fold
    foldTrainImages = reducedImages(trainIdx, :);
    foldTrainLabels = allLabels(trainIdx);
    foldValidImages = reducedImages(validIdx, :);
    foldValidLabels = allLabels(validIdx);
    
    % Train KNN on PCA-reduced training data
    knnModel = fitcknn(foldTrainImages, foldTrainLabels, ...
                       'OptimizeHyperparameters', {'NumNeighbors', 'Distance'}, ...
                       'HyperparameterOptimizationOptions', ...
                       struct('AcquisitionFunctionName', 'expected-improvement-plus'));
    
    % Predict on validation data
    predictedLabels = predict(knnModel, foldValidImages);
    
    % Calculate metrics
    confMat = confusionmat(foldValidLabels, predictedLabels);
    TP = confMat(1, 1);
    FP = confMat(2, 1);
    FN = confMat(1, 2);
    TN = confMat(2, 2);
    
    accuracies(fold) = (TP + TN) / (TP + FP + FN + TN) * 100;
    precisions(fold) = TP / (TP + FP);
    recalls(fold) = TP / (TP + FN);
    specificities(fold) = TN / (TN + FP);
    f1Scores(fold) = 2 * (precisions(fold) * recalls(fold)) / (precisions(fold) + recalls(fold));
    
    % ROC and AUC (approximated using distance-based scores)
    [~, neighborDistances] = knnsearch(foldTrainImages, foldValidImages, 'K', knnModel.NumNeighbors);
    scores = 1 ./ mean(neighborDistances, 2);
    scores = scores / max(scores);
    [~, ~, ~, auc] = perfcurve(foldValidLabels, scores, 1);
    aucs(fold) = auc;
end

%% Display average cross-validation results
fprintf('\nAverage Cross-Validation Results:\n');
fprintf('Accuracy: %.2f%%\n', mean(accuracies));
fprintf('Precision: %.2f\n', mean(precisions));
fprintf('Recall (Sensitivity): %.2f\n', mean(recalls));
fprintf('Specificity: %.2f\n', mean(specificities));
fprintf('F1 Score: %.2f\n', mean(f1Scores));
fprintf('AUC: %.2f\n', mean(aucs));

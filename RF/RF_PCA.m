% Clear workspace and close all figures
clear all;
close all;
clc;

% Set random seed
rng(10);

%% Loading datasets
% training Data
trainFilename = 'face_train.cdataset';
[trainImages, trainLabels] = loadFaceImages2(trainFilename, 1);

% testing Data
testFilename = 'face_test.cdataset';
[testImages, testLabels] = loadFaceImages2(testFilename, 1);

% normalise
testImages = testImages ./ 255.0;
trainImages = trainImages ./ 255.0;

%% Apply PCA
fprintf('Applying PCA...\n');
[eigenVectors, eigenValues, meanX, trainPCAFeatures, ndim] = PrincipalComponentAnalysis(trainImages, 1);

fprintf('PCA complete. Training data dimensionality reduced to %d.\n', ndim);

% Center and project the test data onto PCA space
testCentered = bsxfun(@minus, testImages, meanX);
testPCAFeatures = testCentered * eigenVectors;

%% Hyperparameter Tuning: Find Optimal Number of Trees
fprintf('Finding optimal number of trees...\n');
numTreesRange = 10:10:150; % Range of tree counts to test
accuracyList = zeros(length(numTreesRange), 1); % Store accuracies

% Split data into training and validation sets (80-20 split)
cv = cvpartition(trainLabels, 'Holdout', 0.2);
trainIdx = training(cv);
valIdx = test(cv);

trainFoldFeatures = trainPCAFeatures(trainIdx, :);
trainFoldLabels = trainLabels(trainIdx);
valFoldFeatures = trainPCAFeatures(valIdx, :);
valFoldLabels = trainLabels(valIdx);

for idx = 1:length(numTreesRange)
    numTrees = numTreesRange(idx);
    
    % Train the Random Forest classifier
    forest = TreeBagger(numTrees, trainFoldFeatures, trainFoldLabels, 'Method', 'classification', 'OOBPrediction', 'on');
    
    % Validate the model
    [valPredictions, ~] = predict(forest, valFoldFeatures);
    valPredictions = str2double(valPredictions);
    
    % Compute accuracy
    confMat = confusionmat(valFoldLabels, valPredictions);
    TP = confMat(1, 1);
    FN = confMat(1, 2);
    FP = confMat(2, 1);
    TN = confMat(2, 2);
    accuracy = (TP + TN) / (TP + FP + FN + TN) * 100;
    
    accuracyList(idx) = accuracy; % Store accuracy for this tree count
end

% Find the optimal number of trees
[bestAccuracy, bestIdx] = max(accuracyList);
optimalNumTrees = numTreesRange(bestIdx);

fprintf('Optimal number of trees: %d (Validation Accuracy = %.2f%%)\n', optimalNumTrees, bestAccuracy);

% Visualise accuracy over tree counts
figure;
plot(numTreesRange, accuracyList, '-o', 'LineWidth', 2);
xlabel('Number of Trees');
ylabel('Validation Accuracy (%)');
title('Validation Accuracy vs. Number of Trees');
grid on;

%% Train Final RF Model with Optimal Number of Trees
fprintf('Training final model with %d trees...\n', optimalNumTrees);
RF_PCA_Model = TreeBagger(optimalNumTrees, trainPCAFeatures, trainLabels, 'Method', 'classification', 'OOBPrediction', 'on');

%% Test the Model
fprintf('Testing the RF classifier on test data...\n');
[testPredictions, scores] = predict(RF_PCA_Model, testPCAFeatures);
testPredictions = str2double(testPredictions);

%% Evaluation
confMat = confusionmat(testLabels, testPredictions);
TP = confMat(1, 1);
FN = confMat(1, 2);
FP = confMat(2, 1);
TN = confMat(2, 2);

accuracy = (TP + TN) / (TP + FP + FN + TN) * 100;
precision = TP / (TP + FP + eps);
recall = TP / (TP + FN + eps);
specificity = TN / (TN + FP + eps);
f1Score = 2 * (precision * recall) / (precision + recall + eps);

fprintf('\nPerformance Metrics (Final Model):\n');
fprintf('Accuracy: %.2f%%\n', accuracy);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall (Sensitivity): %.2f\n', recall);
fprintf('Specificity: %.2f\n', specificity);
fprintf('F1 Score: %.2f\n', f1Score);

% Visualise confusion matrix
figure;
confusionchart(testLabels, testPredictions);
title('Confusion Matrix for Final Random Forest Classifier (PCA Features)');

% ROC Curve
fprintf('Plotting ROC curve and calculating AUC...\n');
[rocX, rocY, ~, AUC] = perfcurve(testLabels, scores(:, 2), 1);

figure;
plot(rocX, rocY, 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve for Final RF Classifier (PCA Features)');
legend(sprintf('AUC = %.2f', AUC), 'Location', 'Southeast');
grid on;

% Display AUC
fprintf('Area Under the Curve (AUC): %.2f\n', AUC);

disp('RF model training and evaluation completed.');

% Save the final model
%modelFileName = 'RF_PCA_Model.mat';
%save(modelFileName, 'RF_PCA_Model', 'eigenVectors', 'meanX');
%fprintf('Final model and PCA components saved to %s\n', modelFileName);

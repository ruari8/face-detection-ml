%% Clear workspace and close all figures
clear all;
close all;
clc;
rng(40);

%% Load Training Data
trainFilename = 'face_train.cdataset';
[trainImages, trainLabels] = loadFaceImages2(trainFilename, 1);

testFilename = 'face_test.cdataset';
[testImages, testLabels] = loadFaceImages2(testFilename, 1);

%% Apply PCA
fprintf('Applying PCA...\n');

% Standardize the training data
trainMean = mean(trainImages, 1);
trainStd = std(trainImages, 0, 1);
standardizedTrainImages = (trainImages - trainMean) ./ trainStd;

% Compute PCA
[coeff, score, ~, ~, explained] = pca(standardizedTrainImages);

% Determine number of components to retain (e.g., 95% variance)
cumulativeExplained = cumsum(explained);
numComponents = find(cumulativeExplained >= 95, 1);

% Reduce dimensionality of training data
reducedTrainImages = score(:, 1:numComponents);

% Transform test data using the same PCA model
standardizedTestImages = (testImages - trainMean) ./ trainStd;
reducedTestImages = standardizedTestImages * coeff(:, 1:numComponents);

%% Train KNN on PCA-reduced data
fprintf('Training KNN on PCA-reduced data...\n');

knnModel = fitcknn(reducedTrainImages, trainLabels, ...
                   'OptimizeHyperparameters', {'NumNeighbors', 'Distance'}, ...
                   'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus'));

%% Test KNN
fprintf('Testing KNN...\n');
predictedLabels = predict(knnModel, reducedTestImages);

% Calculate accuracy
accuracy = mean(predictedLabels == testLabels) * 100;
fprintf('Accuracy: %.2f%%\n', accuracy);

%% Evaluation
fprintf('Evaluating KNN model performance...\n');

% Confusion Matrix
confMat = confusionmat(testLabels, predictedLabels);

% Extract True Positives, False Positives, True Negatives, and False Negatives
TP = confMat(1, 1);
FP = confMat(2, 1);
FN = confMat(1, 2);
TN = confMat(2, 2);

% Calculate Metrics
accuracy = (TP + TN) / (TP + FP + FN + TN) * 100;
precision = TP / (TP + FP);
recall = TP / (TP + FN); % Sensitivity
specificity = TN / (TN + FP);
f1Score = 2 * (precision * recall) / (precision + recall);

% Display Metrics
fprintf('Performance Metrics:\n');
fprintf('Accuracy: %.2f%%\n', accuracy);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall (Sensitivity): %.2f\n', recall);
fprintf('Specificity: %.2f\n', specificity);
fprintf('F1 Score: %.2f\n', f1Score);

% Display a Confusion Matrix
figure;
confusionchart(testLabels, predictedLabels);
title('Confusion Matrix for KNN Classifier (PCA Features)');

fprintf('Plotting ROC curve and calculating AUC...\n');

% Extract distance to nearest neighbors for all test points
[~, neighborDistances] = knnsearch(reducedTrainImages, reducedTestImages, ...
                                    'K', knnModel.NumNeighbors);

% Calculate scores as inverse distance (higher score for closer neighbors)
scores = 1 ./ mean(neighborDistances, 2);

% Normalize scores to be between 0 and 1
scores = scores / max(scores);

% Generate the ROC curve data
[rocX, rocY, ~, AUC] = perfcurve(testLabels, scores, 1);

% Plot the ROC curve
figure;
plot(rocX, rocY, 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve for KNN Classifier');
legend(sprintf('AUC = %.2f', AUC), 'Location', 'Southeast');
grid on;

% Display AUC
fprintf('Area Under the Curve (AUC): %.2f\n', AUC);


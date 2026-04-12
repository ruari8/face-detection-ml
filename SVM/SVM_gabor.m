% Clear workspace and close all figures
clear all;
close all;
clc;
rng(40);

% Load an example image to determine its dimensions
imageExample = imread('images/face/1.png');
imageHeight = size(imageExample, 1);
imageWidth = size(imageExample, 2);
fprintf('Detected image size: %d x %d\n', imageHeight, imageWidth);

%% training
% Load the Training Data
trainFilename = 'face_train.cdataset';
[trainImages, trainLabels] = loadFaceImages2(trainFilename, 1);

% Load the Testing Data
testFilename = 'face_test.cdataset';
[testImages, testLabels] = loadFaceImages2(testFilename, 1);

% Normalise the test and train images to [0,1] range
testImages = testImages ./ 255.0;
trainImages = trainImages ./ 255.0;

% Convert images to Gabor feature vectors
[trainFeatures, testFeatures] = gabor_convert_datasets(trainImages, testImages);

% Hyperparameter Optimization with SVM
fprintf('Optimizing SVM hyperparameters...\n');

% Set up the options for hyperparameter optimization
svmModel = fitcsvm(trainFeatures, trainLabels, 'KernelFunction', 'rbf', 'Standardize', true, ...
                    'OptimizeHyperparameters', {'BoxConstraint', 'KernelScale'}, ...
                    'HyperparameterOptimizationOptions', struct(...
                     'AcquisitionFunctionName', 'expected-improvement-plus', ...
                     'ShowPlots', false));

% Display the best-found hyperparameters
bestC = svmModel.ModelParameters.BoxConstraint;
bestKernelScale = svmModel.ModelParameters.KernelScale;
fprintf('Best C (BoxConstraint): %.4f\n', bestC);
fprintf('Best KernelScale (RBF sigma): %.4f\n', bestKernelScale);

%% testing
% Predict Labels and Get Scores for Test Data
fprintf('Testing the SVM classifier on test data...\n');
[predictedLabels, scores] = predict(svmModel, testFeatures);

%% evaluation
% Calculate Performance Metrics

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
title('Confusion Matrix for SVM Classifier (Gabor)');

fprintf('Plotting ROC curve and calculating AUC...\n');

% Generate the ROC curve data
[rocX, rocY, ~, AUC] = perfcurve(testLabels, scores(:, 2), 1);

% Plot the ROC curve
figure;
plot(rocX, rocY, 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve for SVM Classifier');
legend(sprintf('AUC = %.2f', AUC), 'Location', 'Southeast');
grid on;

% Display AUC
fprintf('Area Under the Curve (AUC): %.2f\n', AUC);

disp('SVM model evaluation completed.');

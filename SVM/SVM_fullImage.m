% Clear workspace and close all figures
clear all;
close all;
clc;
rng(40);

%% training
% Step 1: Load the Training Data
trainFilename = 'face_train.cdataset';
[trainImages, trainLabels] = loadFaceImages2(trainFilename, 1, 1);

% Step 2: Extract Features (Full Image)
% No additional processing needed as 'trainImages' already contains the raw pixel values

% Step 3: Hyperparameter Optimization with SVM
fprintf('Optimizing SVM hyperparameters...\n');

% Set up the options for hyperparameter optimization
svmModel = fitcsvm(trainImages, trainLabels, 'KernelFunction', 'rbf', 'Standardize', true, ...
                   'OptimizeHyperparameters', {'BoxConstraint', 'KernelScale'}, ...
                   'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus'));

% Display the best-found hyperparameters
bestC = svmModel.ModelParameters.BoxConstraint;
bestKernelScale = svmModel.ModelParameters.KernelScale;
fprintf('Best C (BoxConstraint): %.4f\n', bestC);
fprintf('Best KernelScale (RBF sigma): %.4f\n', bestKernelScale);

%% testing
% Step 4: Load the Test Data
testFilename = 'face_test.cdataset';
[testImages, testLabels] = loadFaceImages2(testFilename, 1, 1);

% Step 5: Predict Labels and Get Scores for Test Data
fprintf('Testing the SVM classifier on test data...\n');
[predictedLabels, scores] = predict(svmModel, testImages);

%% evaluation
% Step 6: Calculate Performance Metrics

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
title('Confusion Matrix for SVM Classifier (Full Image Features)');

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



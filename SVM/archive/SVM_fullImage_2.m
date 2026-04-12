% Clear workspace and close all figures
clear all;
close all;
clc;
addpath .\SVM-KM\

%% training
% Step 1: Load the Training Data
trainFilename = 'face_train.cdataset';
[trainImages, trainLabels] = loadFaceImages(trainFilename, 1);

% Print initial data statistics
fprintf('Initial training data statistics:\n');
fprintf('Number of samples: %d\n', size(trainImages, 1));
fprintf('Number of face samples: %d\n', sum(trainLabels == 1));
fprintf('Number of non-face samples: %d\n', sum(trainLabels == -1));
fprintf('Feature dimension: %d\n', size(trainImages, 2));

% Data preprocessing
trainImages = trainImages ./ 255.0;

% Step 2: Train the SVM Classifier
fprintf('\nTraining SVM classifier...\n');
svmModel = SVMtraining(trainImages, trainLabels);

%% testing
% Step 3: Load and preprocess the Test Data
testFilename = 'face_test.cdataset';
[testImages, testLabels] = loadFaceImages(testFilename, 1);

% Apply same preprocessing to test data
testImages = testImages ./ 255.0;

% Step 4: Predict Labels
fprintf('Testing the SVM classifier...\n');
classificationResult = zeros(size(testImages,1), 1);
predictionScores = zeros(size(testImages,1), 1);

for i = 1:size(testImages,1)
    [classificationResult(i), predictionScores(i)] = SVMTesting(testHOGFeatures(i, :), svmModel);
end

% Print prediction distribution
fprintf('\nPrediction distribution:\n');
fprintf('Predicted faces: %d\n', sum(classificationResult == 1));
fprintf('Predicted non-faces: %d\n', sum(classificationResult == -1));

%% evaluation
% Step 5: Calculate Performance Metrics
confMat = confusionmat(testLabels, classificationResult);

% Calculate metrics
TP = confMat(1,1); FP = confMat(2,1);
FN = confMat(1,2); TN = confMat(2,2);

accuracy = (TP + TN) / (TP + FP + FN + TN) * 100;
precision = TP / (TP + FP + eps);
recall = TP / (TP + FN + eps);
specificity = TN / (TN + FP + eps);
f1Score = 2 * (precision * recall) / (precision + recall + eps);

fprintf('\nPerformance Metrics:\n');
fprintf('Accuracy: %.2f%%\n', accuracy);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall (Sensitivity): %.2f\n', recall);
fprintf('Specificity: %.2f\n', specificity);
fprintf('F1 Score: %.2f\n', f1Score);

% Visualizations
figure;
confusionchart(testLabels, classificationResult);
title('Confusion Matrix for SVM Classifier');

fprintf('Plotting ROC curve and calculating AUC...\n');

% Generate the ROC curve data
[rocX, rocY, ~, AUC] = perfcurve(testLabels, predictionScores, 1);

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

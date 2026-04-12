% Clear workspace and close all figures
clear all;
close all;
clc;
addpath .\SVM-KM\

% Load an example image to determine its dimensions
imageExample = imread('images/face/1.png');
imageHeight = size(imageExample, 1);
imageWidth = size(imageExample, 2);
fprintf('Detected image size: %d x %d\n', imageHeight, imageWidth);

%% training
% Step 1: Load the Training Data
trainFilename = 'face_train.cdataset';
[trainImages, trainLabels] = loadFaceImages(trainFilename, 1);

% Step 2: Extract HOG Features for Each Image
fprintf('\nExtracting HOG features for training data...\n');
trainHOGFeatures = [];
for i = 1:size(trainImages, 1)
    image = reshape(trainImages(i, :), [imageHeight, imageWidth]);
    feature = hog_feature_vector(image);
    trainHOGFeatures = [trainHOGFeatures; feature];
end

% Step 3: Train the SVM Classifier
fprintf('Training SVM classifier...\n');
svmModel = SVMtraining(trainHOGFeatures, trainLabels);

%% testing
% Step 4: Load and preprocess the Test Data
testFilename = 'face_test.cdataset';
[testImages, testLabels] = loadFaceImages(testFilename, 1);

% Step 5: Extract HOG Features for Each Test Image
fprintf('\nExtracting HOG features for test data...');
testHOGFeatures = [];
for i = 1:size(testImages, 1)
    image = reshape(testImages(i, :), [imageHeight, imageWidth]);
    feature = hog_feature_vector(image);
    testHOGFeatures = [testHOGFeatures; feature];
end

% Step 6: Predict Labels
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
% Step 7: Calculate Performance Metrics
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

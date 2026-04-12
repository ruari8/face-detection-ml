%% Clear workspace and close all figures
clear all;
close all;
clc;
rng(40);

%% Load an example image to determine its dimensions
imageExample = imread('images/face/1.png');
imageHeight = size(imageExample, 1);
imageWidth = size(imageExample, 2);
fprintf('Detected image size: %d x %d\n', imageHeight, imageWidth);

%% Training
% Load the Training Data
trainFilename = 'face_train_6040.cdataset';
[trainImages, trainLabels] = loadFaceImages2(trainFilename, 1);

% Load the Testing Data
testFilename = 'face_test_6040.cdataset';
[testImages, testLabels] = loadFaceImages2(testFilename, 1);

% Normalize the test and train images to [0,1] range
testImages = testImages ./ 255.0;
trainImages = trainImages ./ 255.0;

% Convert images to Gabor feature vectors
[trainFeatures, testFeatures] = gabor_convert_datasets(trainImages, testImages);

% Define range of k values to test
k_values = [3, 5, 7, 10]; % Specify the k values to test
num_k = length(k_values); % Number of k values

% Initialize storage for results
k_results = struct();

% Test different k values
for i = 1:num_k
    k = k_values(i);
    fprintf('\nTesting k = %d\n', k);

    % Train the KNN model with the current k
    knnModel = fitcknn(trainFeatures, trainLabels, ...
                       'NumNeighbors', k, ...
                       'Distance', 'euclidean'); % You can customize the distance metric
    
    % Evaluate on the test set
    [predictedLabels, scores] = predict(knnModel, testFeatures);
    
    % Calculate performance metrics (example: accuracy)
    accuracy = mean(predictedLabels == testLabels) * 100;
    fprintf('Accuracy for k = %d: %.2f%%\n', k, accuracy);
    
    % Store results
    k_results(i).k = k;
    k_results(i).accuracy = accuracy;
end

%% Display the best-found hyperparameters for KNN
bestNumNeighbors = knnModel.NumNeighbors;
bestDistanceMetric = knnModel.Distance;
fprintf('Best Number of Neighbors (k): %d\n', bestNumNeighbors);
fprintf('Best Distance Metric: %s\n', bestDistanceMetric);

%% Testing
% Predict Labels for Test Data
fprintf('Testing the KNN classifier on test data...\n');
[predictedLabels, scores] = predict(knnModel, testFeatures);

%% Evaluation
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
fprintf('Performance Metrics for KNN:\n');
fprintf('Accuracy: %.2f%%\n', accuracy);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall (Sensitivity): %.2f\n', recall);
fprintf('Specificity: %.2f\n', specificity);
fprintf('F1 Score: %.2f\n', f1Score);

% Display a Confusion Matrix
figure;
confusionchart(testLabels, predictedLabels);
title('Confusion Matrix for KNN Classifier (Gabor)');

% Plot ROC Curve and Calculate AUC if Scores are Available
if size(scores, 2) > 1
    fprintf('Plotting ROC curve and calculating AUC...\n');
    [rocX, rocY, ~, AUC] = perfcurve(testLabels, scores(:, 2), 1); % Assuming scores(:, 2) corresponds to positive class
    
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
else
    fprintf('AUC and ROC curve not available for this KNN configuration.\n');
end

disp('KNN model evaluation completed.');

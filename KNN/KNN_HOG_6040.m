%% Clear workspace and close all figures
clear all;
close all;
clc;

%% Load Training Data
trainFilename = 'face_train_6040.cdataset';
[trainImages, trainLabels] = loadFaceImages2(trainFilename, 1);

% Normalize training images
trainImages = trainImages ./ 255.0;

% Load Testing Data
testFilename = 'face_test_6040.cdataset';
[testImages, testLabels] = loadFaceImages2(testFilename, 1);

% Normalize testing images
testImages = testImages ./ 255.0;

%% Convert images to HOG feature vectors
[trainFeatures, testFeatures] = hog_convert_datasets(trainImages, testImages);

%% Train and Evaluate KNN
k_values = [3, 5, 7, 10];
num_k = length(k_values);
results = struct();

for i = 1:num_k
    k = k_values(i);
    fprintf('Testing k = %d\n', k);

    % Train KNN
    knnModel = fitcknn(trainFeatures, trainLabels, ...
                   'NumNeighbors', k, ...
                   'Distance', 'euclidean', ...
                   'DistanceWeight', 'squaredinverse'); 

    
    % Evaluate on the test set
    predictedLabels = predict(knnModel, testFeatures);
    accuracy = mean(predictedLabels == testLabels) * 100;

    fprintf('Accuracy for k = %d: %.2f%%\n', k, accuracy);
    results(i).k = k;
    results(i).accuracy = accuracy;
end

%% Display Results
[~, bestIdx] = max([results.accuracy]);
bestK = results(bestIdx).k;
fprintf('Best k value: %d (Accuracy: %.2f%%)\n', bestK, results(bestIdx).accuracy);
%% Evaluation
fprintf('\nEvaluating the KNN classifier with HOG features...\n');

% Calculate Confusion Matrix
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
fprintf('Performance Metrics for KNN with HOG Features:\n');
fprintf('Accuracy: %.2f%%\n', accuracy);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall (Sensitivity): %.2f\n', recall);
fprintf('Specificity: %.2f\n', specificity);
fprintf('F1 Score: %.2f\n', f1Score);

% Display Confusion Matrix
figure;
confusionchart(testLabels, predictedLabels);
title('Confusion Matrix for KNN Classifier with HOG Features');

[predictedLabels, scores] = predict(knnModel, testFeatures);


% Plot ROC Curve and Calculate AUC if Scores are Available
if size(scores, 2) > 1
    fprintf('Plotting ROC curve and calculating AUC...\n');
    [rocX, rocY, ~, AUC] = perfcurve(testLabels, scores(:, 2), 1); % Assuming scores(:, 2) corresponds to positive class
    
    % Plot the ROC Curve
    figure;
    plot(rocX, rocY, 'LineWidth', 2);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title('ROC Curve for KNN Classifier with HOG Features');
    legend(sprintf('AUC = %.2f', AUC), 'Location', 'Southeast');
    grid on;

    % Display AUC
    fprintf('Area Under the Curve (AUC): %.2f\n', AUC);
else
    fprintf('AUC and ROC curve not available for this KNN configuration.\n');
end

disp('Evaluation of KNN with HOG features completed.');
%% Clear workspace and close all figures
clear all;
close all;
clc;
rng(40);

%% Load Training Data
trainFilename = 'face_train.cdataset';
[trainImages, trainLabels] = loadFaceImages2(trainFilename, 1);

% Normalize training images
trainImages = trainImages ./ 255.0;

%% Convert images to HOG feature vectors
[trainFeatures, ~] = hog_convert_datasets(trainImages, []);

%% Define Cross-Validation Parameters
numFolds = 5; % Number of folds for cross-validation
k_values = [3, 5, 7, 10]; % Specify k values to test
num_k = length(k_values);

% Create cross-validation partition
cvp = cvpartition(trainLabels, 'KFold', numFolds);

% Initialize storage for results
k_results = struct();

%% Perform Cross-Validation
for k_idx = 1:num_k
    k = k_values(k_idx);
    fprintf('\nTesting k = %d with %d-fold cross-validation\n', k, numFolds);
    
    % Initialize metrics for each fold
    accuracies = zeros(numFolds, 1);
    precisions = zeros(numFolds, 1);
    recalls = zeros(numFolds, 1);
    f1Scores = zeros(numFolds, 1);
    aucs = zeros(numFolds, 1);

    for fold = 1:numFolds
        % Get training and validation indices
        trainIdx = training(cvp, fold);
        validIdx = test(cvp, fold);
        
        % Split data into training and validation sets
        foldTrainFeatures = trainFeatures(trainIdx, :);
        foldTrainLabels = trainLabels(trainIdx);
        foldValidFeatures = trainFeatures(validIdx, :);
        foldValidLabels = trainLabels(validIdx);
        
        % Train the KNN model
        knnModel = fitcknn(foldTrainFeatures, foldTrainLabels, ...
                           'NumNeighbors', k, ...
                           'Distance', 'euclidean', ...
                           'DistanceWeight', 'squaredinverse'); % Ensures scores output
        
        % Predict on validation set
        [predictedLabels, scores] = predict(knnModel, foldValidFeatures);
        
        % Calculate confusion matrix
        confMat = confusionmat(foldValidLabels, predictedLabels);
        TP = confMat(1, 1);
        FP = confMat(2, 1);
        FN = confMat(1, 2);
        TN = confMat(2, 2);
        
        % Calculate metrics
        accuracies(fold) = (TP + TN) / (TP + FP + FN + TN) * 100;
        precisions(fold) = TP / (TP + FP);
        recalls(fold) = TP / (TP + FN);
        f1Scores(fold) = 2 * (precisions(fold) * recalls(fold)) / (precisions(fold) + recalls(fold));
        
        % Calculate AUC if scores are available
        if size(scores, 2) > 1
            [~, ~, ~, auc] = perfcurve(foldValidLabels, scores(:, 2), 1);
            aucs(fold) = auc;
        else
            aucs(fold) = NaN; % Set AUC to NaN if scores are unavailable
        end
    end
    
    % Store mean metrics for this k
    k_results(k_idx).k = k;
    k_results(k_idx).mean_accuracy = mean(accuracies);
    k_results(k_idx).std_accuracy = std(accuracies);
    k_results(k_idx).mean_precision = mean(precisions);
    k_results(k_idx).std_precision = std(precisions);
    k_results(k_idx).mean_recall = mean(recalls);
    k_results(k_idx).std_recall = std(recalls);
    k_results(k_idx).mean_f1 = mean(f1Scores);
    k_results(k_idx).std_f1 = std(f1Scores);
    k_results(k_idx).mean_auc = nanmean(aucs); % Handle NaNs
    k_results(k_idx).std_auc = nanstd(aucs);   % Handle NaNs
end

%% Display Results
fprintf('\nResults for different k values (Cross-Validation with HOG):\n');
fprintf('k\tAccuracy(±std)\tPrecision(±std)\tRecall(±std)\tF1(±std)\tAUC(±std)\n');
for i = 1:num_k
    fprintf('%d\t%.2f%%(±%.2f)\t%.2f(±%.2f)\t%.2f(±%.2f)\t%.2f(±%.2f)\t%.2f(±%.2f)\n', ...
        k_results(i).k, ...
        k_results(i).mean_accuracy, k_results(i).std_accuracy, ...
        k_results(i).mean_precision, k_results(i).std_precision, ...
        k_results(i).mean_recall, k_results(i).std_recall, ...
        k_results(i).mean_f1, k_results(i).std_f1, ...
        k_results(i).mean_auc, k_results(i).std_auc);
end

%% Find the Best k Value
[~, bestIdx] = max([k_results.mean_accuracy]);
bestK = k_results(bestIdx).k;
fprintf('\nBest k value: %d (Mean Accuracy: %.2f%% ± %.2f%%)\n', ...
    bestK, k_results(bestIdx).mean_accuracy, k_results(bestIdx).std_accuracy);

disp('Cross-validation for KNN with HOG features completed.');

%% Final Evaluation with Best k
fprintf('\nEvaluating the KNN classifier with HOG features and best k = %d...\n', bestK);

% Re-train the model with the entire training set using the best k
knnModel = fitcknn(trainFeatures, trainLabels, ...
                   'NumNeighbors', bestK, ...
                   'Distance', 'euclidean', ...
                   'DistanceWeight', 'squaredinverse'); % Ensures scores output

% Predict on the full training set (or optionally a separate test set if available)
[predictedLabels, scores] = predict(knnModel, trainFeatures);

% Calculate Confusion Matrix
confMat = confusionmat(trainLabels, predictedLabels);

% Extract True Positives, False Positives, True Negatives, and False Negatives
TP = confMat(1, 1);
FP = confMat(2, 1);
FN = confMat(1, 2);
TN = confMat(2, 2);

% Calculate Performance Metrics
accuracy = (TP + TN) / (TP + FP + FN + TN) * 100;
precision = TP / (TP + FP);
recall = TP / (TP + FN); % Sensitivity
specificity = TN / (TN + FP);
f1Score = 2 * (precision * recall) / (precision + recall);

% Display Metrics
fprintf('\nPerformance Metrics for KNN with HOG Features (Best k = %d):\n', bestK);
fprintf('Accuracy: %.2f%%\n', accuracy);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall (Sensitivity): %.2f\n', recall);
fprintf('Specificity: %.2f\n', specificity);
fprintf('F1 Score: %.2f\n', f1Score);

% Display Confusion Matrix
figure;
confusionchart(trainLabels, predictedLabels);
title(sprintf('Confusion Matrix for KNN Classifier with HOG Features (k = %d)', bestK));

% Plot ROC Curve and Calculate AUC if Scores are Available
if size(scores, 2) > 1
    fprintf('Plotting ROC curve and calculating AUC...\n');
    [rocX, rocY, ~, AUC] = perfcurve(trainLabels, scores(:, 2), 1); % Assuming binary classification with positive class = 1
    
    % Plot the ROC Curve
    figure;
    plot(rocX, rocY, 'LineWidth', 2);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(sprintf('ROC Curve for KNN Classifier with HOG Features (k = %d)', bestK));
    legend(sprintf('AUC = %.2f', AUC), 'Location', 'Southeast');
    grid on;

    % Display AUC
    fprintf('Area Under the Curve (AUC): %.2f\n', AUC);
else
    fprintf('AUC and ROC curve not available for this KNN configuration.\n');
end

disp('Evaluation of KNN with HOG features completed.');

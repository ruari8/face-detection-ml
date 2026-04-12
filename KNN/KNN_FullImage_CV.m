%% Clear workspace and close all figures
clear all;
close all;
clc;
rng(40);
%% Load all data
% Load the complete dataset
[allImages, allLabels] = loadFaceImages2('face_train_6040.cdataset', 1);
[testImages, testLabels] = loadFaceImages2('face_test_6040.cdataset', 1);

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

% Store best parameters across folds
allBestC = zeros(k, 1);
allBestKernelScale = zeros(k, 1);

%% Perform k-fold cross-validation

for fold = 1:k
    fprintf('\nProcessing fold %d/%d...\n', fold, k);
    
    % Get training and validation indices for this fold
    trainIdx = training(cvp, fold);
    validIdx = test(cvp, fold);
    
    % Split data for this fold
    foldTrainImages = allImages(trainIdx, :);
    foldTrainLabels = allLabels(trainIdx);
    foldValidImages = allImages(validIdx, :);
    foldValidLabels = allLabels(validIdx);
    
    % Train KNN with hyperparameter optimization
    fprintf('Optimizing KNN hyperparameters...\n');
    knnModel = fitcknn(foldTrainImages, foldTrainLabels, ...
                       'OptimizeHyperparameters', {'NumNeighbors', 'Distance'}, ...
                       'HyperparameterOptimizationOptions', struct(... 
                        'AcquisitionFunctionName', 'expected-improvement-plus', ...
                        'ShowPlots', false, ...
                        'Verbose', 0));

    % Store best parameters
    allBestK(fold) = knnModel.NumNeighbors; % Numeric value
    allBestDistance{fold} = knnModel.Distance; % Store as cell for string/categorical value
    
    % Predict and evaluate
    [predictedLabels, scores] = predict(knnModel, foldValidImages);
    
    % Calculate metrics for this fold
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
    
    % Calculate AUC (if scores are available for ROC)
    if size(scores, 2) > 1
        [~, ~, ~, auc] = perfcurve(foldValidLabels, scores(:, 2), 1);
        aucs(fold) = auc;
    else
        aucs(fold) = NaN; % If scores are not available, set AUC as NaN
    end
end

%% Train final model using best average parameters and evaluate on test set
fprintf('\nTraining final model with average best parameters...\n');

% Ensure allBestK is populated and calculate average number of neighbors
if isempty(allBestK)
    error('allBestK is empty. Ensure cross-validation has been run and NumNeighbors values are stored.');
end

% Calculate average and round
avgNumNeighbors = round(mean(allBestK)); 

% Determine the most common distance metric
if isempty(allBestDistance)
    error('allBestDistance is empty. Check your cross-validation results.');
end

% Calculate the most common distance
mostCommonDistance = char(mode(categorical(allBestDistance))); 


% Train final KNN model with these parameters
finalModel = fitcknn(allImages, allLabels, ...
                     'NumNeighbors', avgNumNeighbors, ...
                     'Distance', mostCommonDistance, ...
                     'DistanceWeight', 'squaredinverse'); % Use weighted distances if needed

% Evaluate on test set
[predictedLabels, scores] = predict(finalModel, testImages);

% Calculate final test metrics
confMat = confusionmat(testLabels, predictedLabels);
TP = confMat(1, 1);
FP = confMat(2, 1);
FN = confMat(1, 2);
TN = confMat(2, 2);

accuracy = (TP + TN) / (TP + FP + FN + TN) * 100;
precision = TP / (TP + FP);
recall = TP / (TP + FN);
specificity = TN / (TN + FP);
f1Score = 2 * (precision * recall) / (precision + recall);

% Calculate AUC if scores are available
if size(scores, 2) > 1
    [~, ~, ~, AUC] = perfcurve(testLabels, scores(:, 2), 1);
else
    AUC = NaN; % Set AUC as NaN if scores are not available
end

% Display final test results
fprintf('\nFinal Test Set Results:\n');
fprintf('Accuracy: %.2f%%\n', accuracy);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('Specificity: %.2f\n', specificity);
fprintf('F1 Score: %.2f\n', f1Score);
if ~isnan(AUC)
    fprintf('AUC: %.2f\n', AUC);
else
    fprintf('AUC: Not available (scores not provided)\n');
end

% Plot confusion matrix
figure;
confusionchart(testLabels, predictedLabels);
title('Confusion Matrix for Final KNN Model (Test Set)');

% Plot ROC curve if AUC is available
if ~isnan(AUC)
    figure;
    [rocX, rocY, ~, ~] = perfcurve(testLabels, scores(:, 2), 1);
    plot(rocX, rocY, 'LineWidth', 2);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title('ROC Curve for Final KNN Model (Test Set)');
    legend(sprintf('AUC = %.2f', AUC), 'Location', 'Southeast');
    grid on;
end

disp('Cross-validation and final evaluation completed.');


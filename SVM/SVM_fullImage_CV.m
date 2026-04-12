% Clear workspace and close all figures
clear all;
close all;
clc;
rng(40);

%% Load all data
% Load the complete dataset
[allImages, allLabels] = loadFaceImages2('face_train_6040.cdataset', 1, 1);
[testImages, testLabels] = loadFaceImages2('face_test_6040.cdataset', 1, 1);

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
    
    % Train SVM with hyperparameter optimization
    fprintf('Optimizing SVM hyperparameters...\n');
    svmModel = fitcsvm(foldTrainImages, foldTrainLabels, 'KernelFunction', 'rbf', ...
                       'Standardize', true, ...
                       'OptimizeHyperparameters', {'BoxConstraint', 'KernelScale'}, ...
                       'HyperparameterOptimizationOptions', struct(...
                        'AcquisitionFunctionName', 'expected-improvement-plus', ...
                        'ShowPlots', false, ...
                        'Verbose', 0));
    
    % Store best parameters
    allBestC(fold) = svmModel.ModelParameters.BoxConstraint;
    allBestKernelScale(fold) = svmModel.ModelParameters.KernelScale;
    
    % Predict and evaluate
    [predictedLabels, scores] = predict(svmModel, foldValidImages);
    
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
    
    [~, ~, ~, auc] = perfcurve(foldValidLabels, scores(:, 2), 1);
    aucs(fold) = auc;
end

%% Display cross-validation results
fprintf('\nCross-validation Results (mean ± std):\n');
fprintf('Accuracy: %.2f%% ± %.2f%%\n', mean(accuracies), std(accuracies));
fprintf('Precision: %.2f ± %.2f\n', mean(precisions), std(precisions));
fprintf('Recall: %.2f ± %.2f\n', mean(recalls), std(recalls));
fprintf('Specificity: %.2f ± %.2f\n', mean(specificities), std(specificities));
fprintf('F1 Score: %.2f ± %.2f\n', mean(f1Scores), std(f1Scores));
fprintf('AUC: %.2f ± %.2f\n', mean(aucs), std(aucs));

fprintf('\nBest Parameters Across Folds:\n');
fprintf('C (BoxConstraint): %.4f ± %.4f\n', mean(allBestC), std(allBestC));
fprintf('KernelScale: %.4f ± %.4f\n', mean(allBestKernelScale), std(allBestKernelScale));

%% Train final model using best average parameters and evaluate on test set
fprintf('\nTraining final model with average best parameters...\n');

% Train final model with average best parameters
finalModel = fitcsvm(allImages, allLabels, 'KernelFunction', 'rbf', ...
                    'BoxConstraint', mean(allBestC), ...
                    'KernelScale', mean(allBestKernelScale), ...
                    'Standardize', true);

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
[~, ~, ~, AUC] = perfcurve(testLabels, scores(:, 2), 1);

% Display final test results
fprintf('\nFinal Test Set Results:\n');
fprintf('Accuracy: %.2f%%\n', accuracy);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('Specificity: %.2f\n', specificity);
fprintf('F1 Score: %.2f\n', f1Score);
fprintf('AUC: %.2f\n', AUC);

% Plot confusion matrix
figure;
confusionchart(testLabels, predictedLabels);
title('Confusion Matrix for Final SVM Model (Test Set)');

% Plot ROC curve
figure;
[rocX, rocY, ~, ~] = perfcurve(testLabels, scores(:, 2), 1);
plot(rocX, rocY, 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve for Final SVM Model (Test Set)');
legend(sprintf('AUC = %.2f', AUC), 'Location', 'Southeast');
grid on;

disp('Cross-validation and final evaluation completed.');
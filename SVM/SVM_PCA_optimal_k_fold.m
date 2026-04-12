% Clear workspace
clear all;
close all;
clc;
rng(40);

% Define range of k values to test
k_values = [3, 5, 7, 10];
num_k = length(k_values);

% Initialize storage for results
k_results = struct();

for k_idx = 1:num_k
    k = k_values(k_idx);
    fprintf('\nTesting k = %d\n', k);
    
    % Initialize arrays for this k
    accuracies = zeros(k, 1);
    precisions = zeros(k, 1);
    recalls = zeros(k, 1);
    f1Scores = zeros(k, 1);
    aucs = zeros(k, 1);
    
    % Load data
    [trainImages, trainLabels] = loadFaceImages2('face_train_6040.cdataset', 1);
    
    % Create cross-validation partition
    cvp = cvpartition(trainLabels, 'KFold', k);
    
    % Perform k-fold cross-validation
    for fold = 1:k
        % Get training and validation indices for this fold
        trainIdx = training(cvp, fold);
        validIdx = test(cvp, fold);
        
        % Split data
        foldTrainImages = trainImages(trainIdx, :);
        foldTrainLabels = trainLabels(trainIdx);
        foldValidImages = trainImages(validIdx, :);
        foldValidLabels = trainLabels(validIdx);
        
        % Apply PCA
        ndim = 36;
        [eigenVectors, ~, meanX, trainPCAFeatures] = PrincipalComponentAnalysis(foldTrainImages, ndim);
        
        % Train SVM
        svmModel = fitcsvm(trainPCAFeatures, foldTrainLabels, 'KernelFunction', 'rbf', ...
                          'Standardize', true, ...
                          'OptimizeHyperparameters', {'BoxConstraint', 'KernelScale'}, ...
                          'HyperparameterOptimizationOptions', struct(...
                           'AcquisitionFunctionName', 'expected-improvement-plus', ...
                           'ShowPlots', false, ...
                           'Verbose', 0));
        
        % Project validation data
        validCentered = bsxfun(@minus, foldValidImages, meanX);
        validPCAFeatures = validCentered * eigenVectors;
        
        % Predict and evaluate
        [predictedLabels, scores] = predict(svmModel, validPCAFeatures);
        
        % Calculate metrics
        confMat = confusionmat(foldValidLabels, predictedLabels);
        TP = confMat(1, 1);
        FP = confMat(2, 1);
        FN = confMat(1, 2);
        TN = confMat(2, 2);
        
        accuracies(fold) = (TP + TN) / (TP + FP + FN + TN) * 100;
        precisions(fold) = TP / (TP + FP);
        recalls(fold) = TP / (TP + FN);
        f1Scores(fold) = 2 * (precisions(fold) * recalls(fold)) / (precisions(fold) + recalls(fold));
        [~, ~, ~, auc] = perfcurve(foldValidLabels, scores(:, 2), 1);
        aucs(fold) = auc;
    end
    
    % Store results for this k
    k_results(k_idx).k = k;
    k_results(k_idx).mean_accuracy = mean(accuracies);
    k_results(k_idx).std_accuracy = std(accuracies);
    k_results(k_idx).mean_precision = mean(precisions);
    k_results(k_idx).std_precision = std(precisions);
    k_results(k_idx).mean_recall = mean(recalls);
    k_results(k_idx).std_recall = std(recalls);
    k_results(k_idx).mean_f1 = mean(f1Scores);
    k_results(k_idx).std_f1 = std(f1Scores);
    k_results(k_idx).mean_auc = mean(aucs);
    k_results(k_idx).std_auc = std(aucs);
end

% Display results
fprintf('\nResults for different k values:\n');
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

% Plot results
figure;
k_values_plot = [k_results.k];
accuracies_plot = [k_results.mean_accuracy];
std_accuracies = [k_results.std_accuracy];

errorbar(k_values_plot, accuracies_plot, std_accuracies, 'o-', 'LineWidth', 2);
xlabel('k value');
ylabel('Mean Accuracy (%)');
title('Cross-validation Performance vs k Value');
grid on;

disp('Optimal k value assessment completed.');
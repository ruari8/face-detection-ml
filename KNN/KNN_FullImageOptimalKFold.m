%% Clear workspace and close all figures
clear all;
close all;
clc;
rng(40);

%% Initializing

% Define range of k values to test
k_values = [3, 5, 7, 10];
num_k = length(k_values);

% Initialize storage for results
k_results = struct();

%% Testing
for k_idx = 1:num_k
    k = k_values(k_idx);
    fprintf('\nTesting k = %d\n', k);
    
    % Load data
    [trainImages, trainLabels] = loadFaceImages2('face_train_6040.cdataset', 1);
    
    % Create cross-validation partition
    numFolds = 5; % Number of folds for cross-validation
    cvp = cvpartition(trainLabels, 'KFold', numFolds);
    
    % Initialize arrays for this k
    accuracies = zeros(numFolds, 1);
    precisions = zeros(numFolds, 1);
    recalls = zeros(numFolds, 1);
    f1Scores = zeros(numFolds, 1);
    aucs = zeros(numFolds, 1);
    
    % Perform k-fold cross-validation
    for fold = 1:numFolds
        % Get training and validation indices for this fold
        trainIdx = training(cvp, fold);
        validIdx = test(cvp, fold);
        
        % Split data
        foldTrainImages = trainImages(trainIdx, :);
        foldTrainLabels = trainLabels(trainIdx);
        foldValidImages = trainImages(validIdx, :);
        foldValidLabels = trainLabels(validIdx);
        
        % Train KNN
        knnModel = fitcknn(foldTrainImages, foldTrainLabels, ...
                           'NumNeighbors', k, ...
                           'Distance', 'euclidean', ...
                           'DistanceWeight', 'squaredinverse'); % Optional distance weighting
        
        % Predict and evaluate
        [predictedLabels, scores] = predict(knnModel, foldValidImages);
        
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
        
        % Calculate AUC if scores are available
        if size(scores, 2) > 1
            [~, ~, ~, auc] = perfcurve(foldValidLabels, scores(:, 2), 1);
            aucs(fold) = auc;
        else
            aucs(fold) = NaN; % Set AUC as NaN if scores are not available
        end
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
    k_results(k_idx).mean_auc = nanmean(aucs); % Use nanmean to handle NaNs
    k_results(k_idx).std_auc = nanstd(aucs);  % Use nanstd to handle NaNs
end

%% Display results
fprintf('\nResults for different k values (KNN):\n');
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
k_values_plot = [k_results.k]; % Extract k values
accuracies_plot = [k_results.mean_accuracy]; % Mean accuracies for each k
std_accuracies = [k_results.std_accuracy]; % Standard deviations of accuracies

errorbar(k_values_plot, accuracies_plot, std_accuracies, 'o-', 'LineWidth', 2);
xlabel('k value');
ylabel('Mean Accuracy (%)');
title('Cross-validation Performance vs k Value (KNN)');
grid on;

% Find and display the best k value based on accuracy
[bestAccuracy, bestIdx] = max([k_results.mean_accuracy]);
bestK = k_results(bestIdx).k;
fprintf('\nBest performing k value: %d (Accuracy: %.2f%% ± %.2f%%)\n', ...
    bestK, bestAccuracy, k_results(bestIdx).std_accuracy);

disp('Optimal k value assessment completed.');

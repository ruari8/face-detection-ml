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

%% Load the Training Data
trainFilename = 'face_train.cdataset';
[trainImages, trainLabels] = loadFaceImages2(trainFilename, 1);

% Normalize the training images to [0,1] range
trainImages = trainImages ./ 255.0;

% Convert images to Gabor feature vectors
[trainFeatures, ~] = gabor_convert_datasets(trainImages, []);

%% Define Parameters for Cross-Validation
numFolds = 5; % Number of cross-validation folds
k_values = [3, 5, 7, 10]; % k values for KNN
num_k = length(k_values);

% Initialize storage for results
k_results = struct();

% Create cross-validation partition
cvp = cvpartition(trainLabels, 'KFold', numFolds);

%% Perform Cross-Validation
for k_idx = 1:num_k
    k = k_values(k_idx);
    fprintf('\nTesting k = %d\n', k);
    
    % Initialize arrays for metrics
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
                           'Distance', 'euclidean'); % Customize as needed
        
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

% Find the best k value based on accuracy
[~, bestIdx] = max([k_results.mean_accuracy]);
bestK = k_results(bestIdx).k;
fprintf('\nBest k value: %d (Accuracy: %.2f%% ± %.2f%%)\n', ...
    bestK, k_results(bestIdx).mean_accuracy, k_results(bestIdx).std_accuracy);

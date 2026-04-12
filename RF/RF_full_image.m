% This script trains an RF classifier using the full image as a descriptor

% Clear workspace and close all figures
clear all;
close all;
clc;

% Due to random forest randomly selecting different sub samples, set the
% RNG seed so that we get same choice every time
rng(10);

%% Loading datasets
% Load the Training Data
trainFilename = 'face_train.cdataset';
[trainImages, trainLabels] = loadFaceImages(trainFilename, 1);

% Load the Testing Data
testFilename = 'face_test.cdataset';
[testImages, testLabels] = loadFaceImages(testFilename, 1);

% Print initial training data statistics
fprintf('Initial training data statistics:\n');
fprintf('Number of samples: %d\n', size(trainImages, 1));
fprintf('Number of face samples: %d\n', sum(trainLabels == 1));
fprintf('Number of non-face samples: %d\n', sum(trainLabels == -1));
fprintf('Feature dimension: %d\n', size(trainImages, 2));
fprintf('\n');

% Print initial test data statistics
fprintf('Initial test data statistics:\n');
fprintf('Number of samples: %d\n', size(testImages, 1));
fprintf('Number of face samples: %d\n', sum(testLabels == 1));
fprintf('Number of non-face samples: %d\n', sum(testLabels == -1));
fprintf('Feature dimension: %d\n', size(testImages, 2));
fprintf('\n');

% normalise
testImages = testImages ./ 255.0;
trainImages = trainImages ./ 255.0;

%% Evaluate the optimal number of trees
numTreesRange = 10:10:150; % Range of tree counts to test
accuracyList = zeros(length(numTreesRange), 1); % Store accuracies

for idx = 1:length(numTreesRange)
    numTrees = numTreesRange(idx);
    
    % Train the Random Forest classifier
    forest = TreeBagger(numTrees, trainImages, trainLabels, 'Method', 'classification', 'OOBPrediction', 'on');
    
    % Predict labels for the test data
    [testPredictions, ~] = predict(forest, testImages);
    testPredictions = str2double(testPredictions);

    % Confusion Matrix
    confMat = confusionmat(testLabels, testPredictions);

    % Calculate accuracy
    TP = confMat(1,1);
    FN = confMat(1,2);
    FP = confMat(2,1);
    TN = confMat(2,2);
    accuracy = (TP + TN) / (TP + FP + FN + TN) * 100;

    % Store accuracy
    accuracyList(idx) = accuracy;
end

% Display the accuracy table
accuracyTable = table(numTreesRange', accuracyList, 'VariableNames', {'NumberOfTrees', 'Accuracy'});
disp(accuracyTable);

%% Visualise accuracy over the number of trees
figure;
plot(numTreesRange, accuracyList, '-o', 'LineWidth', 2);
xlabel('Number of Trees');
ylabel('Accuracy (%)');
title('Accuracy vs. Number of Trees in Random Forest');
grid on;

% Use the best number of trees for final evaluation
[~, bestIdx] = max(accuracyList);
optimalNumTrees = numTreesRange(bestIdx);

fprintf('\nOptimal Number of Trees: %d\n', optimalNumTrees);
fprintf('Best Accuracy: %.2f%%\n', accuracyList(bestIdx));

% Final RF classifier with optimal number of trees
forest = TreeBagger(optimalNumTrees, trainImages, trainLabels, 'Method', 'classification', 'OOBPrediction', 'on');

% Predict labels for the test data with the optimal model
[testPredictions, scores] = predict(forest, testImages);
testPredictions = str2double(testPredictions);

%% Evaluation
% Confusion Matrix
confMat = confusionmat(testLabels, testPredictions);
TP = confMat(1,1);
FN = confMat(1,2);
FP = confMat(2,1);
TN = confMat(2,2);

accuracy = (TP + TN) / (TP + FP + FN + TN) * 100;
precision = TP / (TP + FP + eps);
recall = TP / (TP + FN + eps);
specificity = TN / (TN + FP + eps);
f1Score = 2 * (precision * recall) / (precision + recall + eps);

fprintf('\nPerformance Metrics (Optimal Model):\n');
fprintf('Accuracy: %.2f%%\n', accuracy);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall (Sensitivity): %.2f\n', recall);
fprintf('Specificity: %.2f\n', specificity);
fprintf('F1 Score: %.2f\n', f1Score);

% Visualisations
figure;
confusionchart(testLabels, testPredictions);
title('Confusion Matrix for Optimal Random Forest Classifier');

fprintf('Plotting ROC curve and calculating AUC...\n');

% Generate the ROC curve data
[rocX, rocY, ~, AUC] = perfcurve(testLabels, scores(:, 2), 1);

% Plot the ROC curve
figure;
plot(rocX, rocY, 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve for Optimal RF Classifier');
legend(sprintf('AUC = %.2f', AUC), 'Location', 'Southeast');
grid on;

% Display AUC
fprintf('Area Under the Curve (AUC): %.2f\n', AUC);

disp('RF model evaluation completed.');

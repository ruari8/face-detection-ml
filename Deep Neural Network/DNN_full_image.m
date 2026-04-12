% Clear workspace and close all figures
clear all;
close all;
clc;

% Set random seed
rng(10);

%% Loading datasets
% Training Data
trainFilename = 'face_train.cdataset';
[trainImages, trainLabels] = loadFaceImages_preprocessed(trainFilename, 1);

% Testing Data
testFilename = 'face_test.cdataset';
[testImages, testLabels] = loadFaceImages_preprocessed(testFilename, 1);

% normalise
testImages = testImages ./ 255.0;
trainImages = trainImages ./ 255.0;

% Reshape data for neural network
imageHeight = 18;
imageWidth = 27;
numChannels = 1;

trainImages = reshape(trainImages', imageHeight, imageWidth, numChannels, []);
testImages = reshape(testImages', imageHeight, imageWidth, numChannels, []);

% Convert labels to categorical (for classification tasks)
trainLabels = categorical(trainLabels);
testLabels = categorical(testLabels);

%% Define Neural Network Architecture
layers = [
    imageInputLayer([imageHeight, imageWidth, numChannels]) % Input layer
    
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];

%% Specify Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 64, ...
    'ValidationData', {testImages, testLabels}, ...
    'ValidationFrequency', 30, ...
    'Verbose', false);

%    'Plots', 'training-progress', ...

%% Train the Neural Network
fprintf('Training the neural network...\n');
net = trainNetwork(trainImages, trainLabels, layers, options);

%% Test
% Get predicted scores for each class
scores = predict(net, testImages);

% Map labels back to original classes
[~, idx] = max(scores, [], 2);
predictedLabels = net.Layers(end).Classes(idx); % Map to original classes [-1, 1]

if ~iscategorical(predictedLabels)
    predictedLabels = categorical(predictedLabels);
end

predictedLabels = categorical(predictedLabels);
testLabels = categorical(testLabels);

%% Evaluation
% Get Confusion Matrix
confMat = confusionmat(testLabels, predictedLabels);

% Calculate performance metrics
TP = confMat(1, 1);
FN = confMat(1, 2);
FP = confMat(2, 1);
TN = confMat(2, 2);

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

% confusion matrix
figure;
confusionchart(testLabels, predictedLabels);
title('Confusion Matrix for Neural Network Model');

%% Plot ROC Curve and Calculate AUC
fprintf('Plotting ROC curve and calculating AUC...\n');

[rocX, rocY, ~, AUC] = perfcurve(testLabels, scores(:, 2), 1);

% Plot the ROC curve
figure;
plot(rocX, rocY, 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve for Neural Network Model');
legend(sprintf('AUC = %.2f', AUC), 'Location', 'Southeast');
grid on;

% Display AUC
fprintf('Area Under the Curve (AUC): %.2f\n', AUC);

disp('Model evaluation completed.');

%% Save the Model
modelFileName = 'NeuralNetModel.mat';
save(modelFileName, 'net');
fprintf('Trained neural network model saved to %s\n', modelFileName);
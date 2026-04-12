% Load Dataset
[images, labels] = loadFaceImages('face_train.cdataset');

imageHeight = 18;
imageWidth = 27;

% Extract Features (Gabor features)
numImages = size(images, 1);
sampleImage = reshape(images(1, :), [imageHeight, imageWidth]);
gaborFeatureLength = length(gabor_feature_vector(sampleImage));
gaborFeatures = zeros(numImages, gaborFeatureLength);

for i = 1:numImages
    % Reshape each image to the correct dimensions
    reshapedImage = reshape(images(i, :), [imageHeight, imageWidth]);
    % Compute Gabor features for the reshaped image
    gaborFeatures(i, :) = gabor_feature_vector(reshapedImage);
end

% Train KNN
knnModel = fitcknn(gaborFeatures, labels, 'NumNeighbors', 3);  % Adjust k as needed

% Testing and Evaluation
[testImages, testLabels] = loadFaceImages('face_test.cdataset');
numTestImages = size(testImages, 1);
testFeatures = zeros(numTestImages, gaborFeatureLength);

for i = 1:numTestImages
    % Reshape each test image
    reshapedTestImage = reshape(testImages(i, :), [imageHeight, imageWidth]);
    % Compute Gabor features for the reshaped test image
    testFeatures(i, :) = gabor_feature_vector(reshapedTestImage);
end

% Predict and evaluate
predictions = predict(knnModel, testFeatures);
accuracy = sum(predictions == testLabels) / numel(testLabels);
disp(['Accuracy: ', num2str(accuracy)]);

% Calculate confusion matrix
confMat = confusionmat(testLabels, predictions);

% Extract TP, FP, TN, FN for binary classification
TP = confMat(1,1);
FN = confMat(1,2);
FP = confMat(2,1);
TN = confMat(2,2);

% Calculate precision, recall, specificity, and F1 score
precision = TP / (TP + FP);
recall = TP / (TP + FN);
specificity = TN / (TN + FP);
f1_score = 2 * (precision * recall) / (precision + recall);

% Display calculated metrics
disp(['Precision: ', num2str(precision)]);
disp(['Recall: ', num2str(recall)]);
disp(['Specificity: ', num2str(specificity)]);
disp(['F1 Score: ', num2str(f1_score)]);

% Calculate ROC and AUC
[~, scores] = predict(knnModel, testFeatures);
[X, Y, T, AUC] = perfcurve(testLabels, scores(:, 2), 1); % Assuming '1' is the positive class

% Display AUC
disp(['AUC: ', num2str(AUC)]);

% Plot ROC curve
figure;
plot(X, Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');
grid on;


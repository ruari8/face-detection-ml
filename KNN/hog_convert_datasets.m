function [trainFeatures, testFeatures] = hog_convert_datasets(trainImages, testImages)
    % Extract HOG features for training images
    numTrain = size(trainImages, 1);
    imgSize = [27, 18]; % Adjust dimensions as needed
    hogFeatureSize = length(extractHOGFeatures(reshape(trainImages(1, :), imgSize))); 
    trainFeatures = zeros(numTrain, hogFeatureSize);
    
    for i = 1:numTrain
        img = reshape(trainImages(i, :), imgSize);
        trainFeatures(i, :) = extractHOGFeatures(img);
    end

    % Extract HOG features for testing images
    numTest = size(testImages, 1);
    testFeatures = zeros(numTest, hogFeatureSize);
    
    for i = 1:numTest
        img = reshape(testImages(i, :), imgSize);
        testFeatures(i, :) = extractHOGFeatures(img);
    end
end

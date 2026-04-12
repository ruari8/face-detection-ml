function [trainFeatures, testFeatures] = gabor_convert_datasets(trainImages, testImages)
    
    % function to convert the training and test datasets to gabor features
    numTrainSamples = size(trainImages, 1);
    numTestSamples = size(testImages, 1);

    featureLength = 19440;  
    trainFeatures = zeros(numTrainSamples, featureLength);
    testFeatures = zeros(numTestSamples, featureLength);
    
    % Convert training images to Gabor feature vectors
    for i = 1:numTrainSamples
        image = reshape(trainImages(i, :), [18, 27]);
        trainFeatures(i, :) = gabor_feature_vector(image);
    end
    
    % Convert test images to Gabor feature vectors
    for i = 1:numTestSamples
        image = reshape(testImages(i, :), [18, 27]);
        testFeatures(i, :) = gabor_feature_vector(image);
    end
end

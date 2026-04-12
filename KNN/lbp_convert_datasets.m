function [trainFeatures, testFeatures] = lbp_convert_datasets(trainImages, testImages)
    % Parameters for LBP
    numPoints = 8; % Number of circularly symmetric neighbor set points
    radius = 1;    % Radius for the circular neighborhood

    % Extract LBP features for training images
    numTrain = size(trainImages, 1);
    trainFeatures = zeros(numTrain, 59); % Uniform LBP has 59 bins
    for i = 1:numTrain
        img = reshape(trainImages(i, :), 27, 18); % Assuming 27x18 image size
        lbp = extractLBPFeatures(img, 'NumNeighbors', numPoints, 'Radius', radius, ...
                                 'Upright', true, 'Normalization', 'L2');
        trainFeatures(i, :) = lbp;
    end

    % Extract LBP features for testing images
    numTest = size(testImages, 1);
    testFeatures = zeros(numTest, 59); % Uniform LBP has 59 bins
    for i = 1:numTest
        img = reshape(testImages(i, :), 27, 18); % Assuming 27x18 image size
        lbp = extractLBPFeatures(img, 'NumNeighbors', numPoints, 'Radius', radius, ...
                                 'Upright', true, 'Normalization', 'L2');
        testFeatures(i, :) = lbp;
    end
end


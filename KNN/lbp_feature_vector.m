function lbpFeatures = lbp_feature_vector(image)
    % Ensure the image is grayscale
    if size(image, 3) == 3
        image = rgb2gray(image);
    end

    % Resize image if too small for default LBP settings
    minSize = 32;  % Minimum dimension requirement for radius 1
    if size(image, 1) < minSize || size(image, 2) < minSize
        image = imresize(image, [minSize, minSize]);
    end

    % Apply LBP with radius 1 and 8 neighbors
    lbpImage = extractLBPFeatures(image, 'Upright', false, 'Radius', 1, 'NumNeighbors', 8);

    % Return the LBP image as a feature vector
    lbpFeatures = lbpImage;
end

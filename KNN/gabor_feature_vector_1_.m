function gaborFeatures = gabor_feature_vector(image)
    % Convert to grayscale if the image has multiple color channels
    if size(image, 3) == 3
        image = rgb2gray(image);
    end

    % Ensure the image is in uint8 format
    image = adapthisteq(image, 'Numtiles', [8 3]);
    image = im2uint8(image);

    % Define Gabor filter parameters
    wavelength = 4;  % Adjust as necessary
    orientation = 0;  % Adjust as necessary

    % Create Gabor filter
    gaborArray = gabor(wavelength, orientation);

    % Apply Gabor filter
    gaborMag = imgaborfilt(image, gaborArray);

    % Flatten Gabor magnitude output to form feature vector
    gaborFeatures = gaborMag(:);
end

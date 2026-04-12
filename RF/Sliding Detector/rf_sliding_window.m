% Improved Sliding Window Detector Script

function [finalDetections, allScores] = sliding_window(image, model, confidenceThreshold)
    % Convert image to grayscale if needed
    if size(image, 3) == 3
        grayImage = rgb2gray(image);
    else
        grayImage = image;
    end

    % Define improved parameters
    windowSize = [27, 18];  % Base window size
    scales = linspace(1.2, 0.8, 8);
    
    % Adaptive step size based on window size
    baseStepSize = 3;
    padding = 10;  % Padding around image edges
    
    % Initialize detections array [x, y, scale, score]
    detections = [];
    allScores = [];  % Store all scores for visualization
    
    % Add padding to image
    paddedImage = padarray(grayImage, [padding padding], 'replicate');
    
    fprintf('Running sliding window detector with %d scales...\n', length(scales));
    
    % Loop through each scale
    for scaleIdx = 1:length(scales)
        scale = scales(scaleIdx);
        fprintf('Processing scale %.2f (%d/%d)\n', scale, scaleIdx, length(scales));
        
        % Resize image at current scale
        resizedImage = imresize(paddedImage, scale);
        [resizedHeight, resizedWidth] = size(resizedImage);
        
        % Adaptive step size based on current scale
        currentStepSize = max(1, round(baseStepSize * scale));
        
        % Loop through sliding windows
        for y = 1:currentStepSize:(resizedHeight - windowSize(1))
            for x = 1:currentStepSize:(resizedWidth - windowSize(2))
                % Extract window region
                window = resizedImage(y:y+windowSize(1)-1, x:x+windowSize(2)-1);
                
                % Skip if window size is incorrect
                if ~isequal(size(window), windowSize([1 2]))
                    continue;
                end
                
                % Extract features
                features = hog_feature_vector(window);
                
                % Prediction and scores using TreeBagger
                [prediction, scores] = predict(model, features);
                predictedLabel = str2double(prediction);
                
                % Check if the prediction is a face (label = 1) and the score exceeds the threshold
                if predictedLabel == 1 && scores(:, 2) > confidenceThreshold
                    % Adjust coordinates to account for padding and scale
                    adjustedX = max(1, round((x - padding) / scale));
                    adjustedY = max(1, round((y - padding) / scale));
                    detections = [detections; adjustedX, adjustedY, scale, scores(:, 2)];
                    allScores = [allScores; scores(:, 2)];
                end
            end
        end
    end
    
    fprintf('Found %d initial detections\n', size(detections, 1));
    
    % Apply Non-Maximum Suppression with improved threshold
    if ~isempty(detections)
        overlapThreshold = 0.15;  % Increased NMS threshold
        finalDetections = applyNMS(detections, overlapThreshold, windowSize);
        fprintf('Reduced to %d detections after NMS\n', size(finalDetections, 1));
    else
        finalDetections = [];
        fprintf('No detections found\n');
    end
end
function [finalDetections, allScores] = sliding_window_pca(image, model, eigenVectors, meanX, confidenceThreshold)
    % Convert image to grayscale if needed
    if size(image, 3) == 3
        grayImage = rgb2gray(image);
    else
        grayImage = image;
    end

    % Parameters
    windowSize = [27, 18]; % Base window size
    scales = linspace(1.2, 0.8, 8); % Scale factors
    baseStepSize = 3; % Base step size
    padding = 10; % Image padding

    % Pad image
    paddedImage = padarray(grayImage, [padding padding], 'replicate');

    % Initialize detections and scores
    detections = zeros(10000, 4); % Preallocate large enough for efficiency
    allScores = zeros(10000, 1); % Preallocate for scores
    detectionIdx = 0;

    % Parallel loop over scales
    fprintf('Running sliding window detector with %d scales...\n', length(scales));
    for scaleIdx = 1:length(scales)
        scale = scales(scaleIdx);
        fprintf('Processing scale %.2f (%d/%d)\n', scale, scaleIdx, length(scales));

        % Resize image for the current scale
        resizedImage = imresize(paddedImage, scale);
        [resizedHeight, resizedWidth] = size(resizedImage);

        % Step size at the current scale
        currentStepSize = max(1, round(baseStepSize * scale));

        % Sliding window
        for y = 1:currentStepSize:(resizedHeight - windowSize(1))
            for x = 1:currentStepSize:(resizedWidth - windowSize(2))
                % Extract and process window
                window = resizedImage(y:y+windowSize(1)-1, x:x+windowSize(2)-1);
                windowVector = double(reshape(window, 1, []));
                features = (windowVector - meanX) * eigenVectors;

                % Prediction
                [prediction, scores] = predict(model, features);
                predictedLabel = str2double(prediction);

                % Check for face detection
                if predictedLabel == 1 && scores(:, 2) > confidenceThreshold
                    adjustedX = max(1, round((x - padding) / scale));
                    adjustedY = max(1, round((y - padding) / scale));
                    
                    % Store results
                    detectionIdx = detectionIdx + 1;
                    detections(detectionIdx, :) = [adjustedX, adjustedY, scale, scores(:, 2)];
                    allScores(detectionIdx) = scores(:, 2);
                end
            end
        end
    

    % Trim unused preallocated space
    detections = detections(1:detectionIdx, :);
    allScores = allScores(1:detectionIdx);

    % Apply Non-Maximum Suppression
    if ~isempty(detections)
        overlapThreshold = 0.15;
        finalDetections = applyNMS(detections, overlapThreshold, windowSize);
        fprintf('Reduced to %d detections after NMS\n', size(finalDetections, 1));
    else
        finalDetections = [];
        fprintf('No detections found\n');
    end
end

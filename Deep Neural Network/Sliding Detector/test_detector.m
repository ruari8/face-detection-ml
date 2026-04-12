% Script to run and visualize the detector results
clear all;
close all;
clc;

enableHogFeatures = 0;

% Load model
modelFileName = 'NeuralNetModel.mat';
load(modelFileName);
save(modelFileName, 'net');
fprintf('Model loaded %s', modelFileName);


% Test images
testImages = {'im1.jpg', 'im2.jpg', 'im3.jpg', 'im4.jpg'};

% Create a figure for the grid layout of all images
figure('Name', 'All Detections Grid View');

% Store processed images and their detections for the grid view
gridImages = cell(1, 4);
gridDetections = cell(1, 4);

% Process each test image
for i = 1:length(testImages)
    % Load image
    image = imread(testImages{i});
    
    % Run detector
    [detections, scores] = dnn_sliding_window(image, net, 0.65, enableHogFeatures);

    % Store for grid view
    gridImages{i} = image;
    gridDetections{i} = detections;
    
    % Create individual figure for each image
    figure('Name', sprintf('Results for %s', testImages{i}));
    
    % Show original image with detections
    subplot(1,2,1);
    imshow(image);
    hold on;
    
    % Draw detection boxes
    if ~isempty(detections)
        for j = 1:size(detections, 1)
            x = detections(j, 1);
            y = detections(j, 2);
            scale = detections(j, 3);
            score = detections(j, 4);
            
            % Calculate window size at this scale
            width = round(18 / scale);   % windowSize(2)
            height = round(27 / scale);  % windowSize(1)
            
            % Draw rectangle and score
            rectangle('Position', [x, y, width, height], ...
                     'EdgeColor', 'r', 'LineWidth', 2);
            text(x, y-5, sprintf('%.2f', score), ...
                 'Color', 'red', 'FontSize', 8);
        end
    end
    title('Detections with Bounding Boxes');
    hold off;
    
    % Plot score distribution
    if ~isempty(scores)
        subplot(1,2,2);
        histogram(scores, 20, 'Normalization', 'probability');
        title('Detection Score Distribution');
        xlabel('Score');
        ylabel('Probability');
    end
end

% Return to the grid figure and plot all images
figure(1);
for i = 1:4
    subplot(2,2,i);
    imshow(gridImages{i});
    hold on;
    
    % Draw detection boxes
    if ~isempty(gridDetections{i})
        for j = 1:size(gridDetections{i}, 1)
            x = gridDetections{i}(j, 1);
            y = gridDetections{i}(j, 2);
            scale = gridDetections{i}(j, 3);
            score = gridDetections{i}(j, 4);
            
            % Calculate window size at this scale
            width = round(18 / scale);
            height = round(27 / scale);
            
            % Draw rectangle and score
            rectangle('Position', [x, y, width, height], ...
                     'EdgeColor', 'r', 'LineWidth', 2);
            text(x, y-5, sprintf('%.2f', score), ...
                 'Color', 'red', 'FontSize', 8);
        end
    end
    title(sprintf('Image %d', i));
    hold off;
end

sgtitle('All Detections Overview');
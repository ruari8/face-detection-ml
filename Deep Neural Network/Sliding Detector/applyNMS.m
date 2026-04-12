function finalDetections = applyNMS(detections, overlapThreshold, windowSize)
    % If no detections, return empty array
    if isempty(detections)
        finalDetections = [];
        return;
    end
    
    % Sort detections by confidence score (descending)
    [~, sortedIndices] = sort(detections(:,4), 'descend');
    detections = detections(sortedIndices,:);
    
    % Initialize array to keep track of which detections to keep
    keep = true(1, size(detections,1));
    
    % Loop through detections
    for i = 1:size(detections,1)
        if ~keep(i)
            continue;
        end
        
        % Get current detection
        det_i = detections(i,:);
        
        % Compare with all remaining detections
        for j = i+1:size(detections,1)
            if ~keep(j)
                continue;
            end
            
            % Get comparison detection
            det_j = detections(j,:);
            
            % Calculate overlap using IoU (Intersection over Union)
            % Extract coordinates and dimensions for first detection
            x1 = det_i(1);
            y1 = det_i(2);
            s1 = det_i(3);
            w1 = round(windowSize(2) / s1);
            h1 = round(windowSize(1) / s1);
            
            % Extract coordinates and dimensions for second detection
            x2 = det_j(1);
            y2 = det_j(2);
            s2 = det_j(3);
            w2 = round(windowSize(2) / s2);
            h2 = round(windowSize(1) / s2);
            
            % Calculate intersection coordinates
            intersect_x1 = max(x1, x2);
            intersect_y1 = max(y1, y2);
            intersect_x2 = min(x1 + w1, x2 + w2);
            intersect_y2 = min(y1 + h1, y2 + h2);
            
            % Calculate intersection area
            intersect_width = max(0, intersect_x2 - intersect_x1);
            intersect_height = max(0, intersect_y2 - intersect_y1);
            intersection_area = intersect_width * intersect_height;
            
            % Calculate union area
            area1 = w1 * h1;
            area2 = w2 * h2;
            union_area = area1 + area2 - intersection_area;
            
            % Calculate IoU
            iou = intersection_area / union_area;
            
            % If overlap exceeds threshold, suppress the lower scoring detection
            if iou > overlapThreshold
                keep(j) = false;
            end
        end
    end
    
    % Return only the kept detections
    finalDetections = detections(keep,:);
end
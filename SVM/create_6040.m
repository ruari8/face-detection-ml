% Clear workspace
% Note: ensure matlab working directory is currently inside this folder
% otherwise Matlab wont be able to access the images directory
clear all;
clc;
rng(40); % For reproducibility

% Read face and non-face image directories
faceDir = dir('images/face/*.png');
nonfaceDir = dir('images/non-face/*.png');

% Extract actual image numbers
nonfaceNums = [];
for i = 1:length(nonfaceDir)
    [~, name, ~] = fileparts(nonfaceDir(i).name);
    nonfaceNums = [nonfaceNums str2double(name)];
end
nonfaceNums = sort(nonfaceNums); % Get actual available numbers

% Create arrays of all image indices and labels
allIndices = [];
allTypes = [];  % 1 for face, 2 for non-face

% Add face images (1-69)
for i = 1:length(faceDir)
    allIndices = [allIndices i];
    allTypes = [allTypes 1];
end

% Add only existing non-face images
for i = 1:length(nonfaceNums)
    allIndices = [allIndices nonfaceNums(i)];
    allTypes = [allTypes 2];
end

% Randomly shuffle all indices together
numTotal = length(allIndices);
shuffleIdx = randperm(numTotal);
allIndices = allIndices(shuffleIdx);
allTypes = allTypes(shuffleIdx);

% Split into train/test
splitPoint = round(0.6 * numTotal);
trainIndices = allIndices(1:splitPoint);
trainTypes = allTypes(1:splitPoint);
testIndices = allIndices(splitPoint+1:end);
testTypes = allTypes(splitPoint+1:end);

% Create training dataset file
fid = fopen('face_train_6040.cdataset', 'w');
fprintf(fid, 'CroppedImageDatabase\n');
fprintf(fid, '18 27\n');
fprintf(fid, '%d\n', length(trainIndices));

% Write face images first (ordered)
faceIdxs = sort(trainIndices(trainTypes == 1));
for i = faceIdxs
    fprintf(fid, '+1 images/face/%d.png\n', i);
end

% Write non-face images second (ordered)
nonfaceIdxs = sort(trainIndices(trainTypes == 2));
for i = nonfaceIdxs
    fprintf(fid, '-1 images/non-face/%d.png\n', i);
end
fclose(fid);

% Create testing dataset file
fid = fopen('face_test_6040.cdataset', 'w');
fprintf(fid, 'CroppedImageDatabase\n');
fprintf(fid, '18 27\n');
fprintf(fid, '%d\n', length(testIndices));

% Write face images first (ordered)
faceIdxs = sort(testIndices(testTypes == 1));
for i = faceIdxs
    fprintf(fid, '+1 images/face/%d.png\n', i);
end

% Write non-face images second (ordered)
nonfaceIdxs = sort(testIndices(testTypes == 2));
for i = nonfaceIdxs
    fprintf(fid, '-1 images/non-face/%d.png\n', i);
end
fclose(fid);

% Display statistics
fprintf('Created new dataset files:\n');
fprintf('Training: face_train_6040.cdataset (%d images)\n', length(trainIndices));
fprintf('Testing: face_test_6040.cdataset (%d images)\n', length(testIndices));

fprintf('\nTraining set composition:\n');
fprintf('Faces: %d\n', sum(trainTypes == 1));
fprintf('Non-faces: %d\n', sum(trainTypes == 2));

fprintf('\nTesting set composition:\n');
fprintf('Faces: %d\n', sum(testTypes == 1));
fprintf('Non-faces: %d\n', sum(testTypes == 2));

% Display missing non-face numbers
fprintf('\nNon-face images used: ');
fprintf('%d ', nonfaceNums);
fprintf('\n');
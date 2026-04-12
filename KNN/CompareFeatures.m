% Clear workspace and close all figures
clearvars;
close all;
clc;

% List of all scripts to run
scripts = {
    'KNN_FullImage.m', ...
    'KNN_FullImage6040.m', ...
    'KNN_FullImage_CV.m', ...
    'KNN_PCA.m', ...
    'KNN_PCA_6040.m', ...
    'KNN_PCA_CV.m', ...
    'KNN_HOG.m', ...
    'KNN_HOG_6040.m', ...
    'KNN_HOG_CV.m', ...
    'KNN_LBP.m', ...
    'KNN_LBP_6040.m', ...
    'KNN_LBP_CV.m', ...
    'KNN_gabor.m', ...
    'KnnGabor6040.m', ...
    'KnnGaborCV.m'
};

% Open a log file to write output and errors
logFileName = 'RunScriptsLog.txt';
logFile = fopen(logFileName, 'a'); % Open in append mode to avoid overwriting
if logFile == -1
    error('Cannot open log file for writing.');
end

% Write log header
fprintf(logFile, 'Script Execution Log\n');
fprintf(logFile, '====================\n\n');

% Loop through each script and run it
for i = 1:length(scripts)
    scriptName = scripts{i};
    fprintf('Running script: %s\n', scriptName);
    fprintf(logFile, 'Running script: %s\n', scriptName);
    
    try
        % Redirect command window output to a temporary diary file
        diary('tempDiary.txt'); % Create a temporary diary file
        diary on;              % Start capturing output
        
        % Run the script
        run(scriptName);
        
        % Stop capturing output and append to the log file
        diary off;
        if exist('tempDiary.txt', 'file')
            tempDiaryContent = fileread('tempDiary.txt');
            fprintf(logFile, '%s\n', tempDiaryContent);
            delete('tempDiary.txt'); % Remove temporary diary file
        end
        
        % Log success message
        fprintf(logFile, '%s completed successfully.\n\n', scriptName);
    catch ME
        % Stop capturing output and handle errors
        diary off;
        if exist('tempDiary.txt', 'file')
            tempDiaryContent = fileread('tempDiary.txt');
            fprintf(logFile, '%s\n', tempDiaryContent);
            delete('tempDiary.txt'); % Remove temporary diary file
        end
        
        % Log error message
        fprintf(logFile, 'Error in %s: %s\n\n', scriptName, ME.message);
    end
end

% Close the log file
fclose(logFile);

% Display completion message
disp('All scripts have been executed. Check RunScriptsLog.txt for details.');

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

% Define method names corresponding to the scripts
methodNames = {
    'Full Image', 'Full Image 60/40', 'Full Image CV', ...
    'PCA', 'PCA 60/40', 'PCA CV', ...
    'HOG', 'HOG 60/40', 'HOG CV', ...
    'LBP', 'LBP 60/40', 'LBP CV', ...
    'Gabor', 'Gabor 60/40', 'Gabor CV'
};

% Create an empty results table
resultsTable = table(methodNames', nan(length(scripts), 1), nan(length(scripts), 1), ...
    nan(length(scripts), 1), nan(length(scripts), 1), nan(length(scripts), 1), ...
    nan(length(scripts), 1), ...
    'VariableNames', {'Method', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1_Score', 'AUC'});

% Loop through each script and run it
for i = 1:length(scripts)
    fprintf('Running script: %s\n', scripts{i});
    
    try
        % Run the script
        run(scripts{i});
        
        % Collect metrics (assumes the script defines these variables)
        if exist('accuracy', 'var'), resultsTable.Accuracy(i) = accuracy; end
        if exist('precision', 'var'), resultsTable.Precision(i) = precision; end
        if exist('recall', 'var'), resultsTable.Recall(i) = recall; end
        if exist('specificity', 'var'), resultsTable.Specificity(i) = specificity; end
        if exist('f1Score', 'var'), resultsTable.F1_Score(i) = f1Score; end
        if exist('AUC', 'var'), resultsTable.AUC(i) = AUC; end
        
        % Clear temporary variables to avoid contamination
        clear accuracy precision recall specificity f1Score AUC;
    catch ME
        % Log the error and continue to the next script
        fprintf('Error running %s: %s\n', scripts{i}, ME.message);
    end
    
    % Write the updated results table to a CSV file
    writetable(resultsTable, 'ResultsSummary.csv');
    fprintf('Intermediate results saved to ResultsSummary.csv\n');
end

% Display the final results table
disp(resultsTable);
fprintf('Final results saved to ResultsSummary.csv\n');

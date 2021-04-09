%%%%%%%%%%%%%%
%
% This file loads the settings (generated in the config.m file) and, 
% for each database, it generates a set with 10 trained models, 
% calculating the error of each one of them in the validation set.
%
% This process is executed 20 times
%
%%%%%%%%%%%%%%
load('MINE/results/configs')

numberRegressors = length(modelsUsed);

for i=1:length(datasetsUsed)
    
     [data, labels] = loadData(sprintf('data/%s.data', datasetsUsed{i}));

     %%%%%%%%%%%%%%%%%%%%%
     % Data Normalization
     %%%%%%%%%%%%%%%%%%%%%
     labelsTemp = normalizeMatrix(labels);
     dataTemp = normalizeMatrix(data);
     %%%%%%%%%%%%%%%%%%%%%
     %%%%%%%%%%%%%%%%%%%%%
     
     data = [labelsTemp,dataTemp]; % Merge data and labels 
        
     errorsBases = zeros(executions,numberRegressors);

     for jBases = 1 : executions
        
        fprintf('Dataset: %s\nExecution: %d\n',datasetsUsed{i},jBases);
       
        [trainIndex, validIndex, testIndex] = buildFolds(labelsTemp, folds, 7, 2, 1);
        
        errorsRegressors = zeros(folds,numberRegressors);

        poolAll = cell(1,folds);
        
        for iFolds = 1 : folds

            train = logical(trainIndex(:,iFolds));
            test = logical(testIndex(:,iFolds));        
            valid = logical(validIndex(:,iFolds));

            dataTrain = data(train,:);
            dataTest = data(test,:);
            dataValid = data(valid,:);

            % Training
            poolAll{1,iFolds} = EnsembleGeneration10(numberRegressors, dataTrain, 1);

            % Validation Set to test the models
            errorsRegressors(iFolds,:) = calculateErrors(poolAll{1,iFolds},dataValid, modelsUsed);

        end
        
        errorsBases(jBases,:) = mean(errorsRegressors);
        
        filenameMAT = strcat(resultsPathPhase1, datasetsUsed{i}, '-', num2str(jBases), '-10.mat');
        save(filenameMAT,'errorsRegressors','poolAll','trainIndex','validIndex','testIndex','data');      
     end
     [~,indexBetterModel] = min(mean(errorsBases));
     
     filenameMATFinal = strcat(resultsPathPhase1, datasetsUsed{i}, '- FINAL -','10.mat');
     save(filenameMATFinal,'errorsBases','indexBetterModel');
end
%%%%%%%%%%%%%%%%%%%
%
% Function that generates the ensemble with 100 regressors to be used in
% the Phase 3 and Phase 4
%
%%%%%%%%%%%%%%%%%%%
function [] = runGenerateEnsemble100(dataset,exec,regressorName,neighbors,neurons)

    configFile = 'MINE/results/configs';
    load(configFile,'resultsPathPhase1');
    load(configFile,'resultsPathPhase2');
    load(configFile,'folds');

    fprintf('\nDataset: %s-%d\n', dataset,exec);
    
    filenameExec = strcat(resultsPathPhase1, dataset, '-', num2str(exec), '-10.mat');
    load(filenameExec,'trainIndex','validIndex','data'); % We only load the Training and Validation indexes
    
    warning('off','all');
    
    pool = cell(folds,1);
    unique = cell(folds,1);

    for i=1:folds
        train = logical(trainIndex(:,i));
        valid = logical(validIndex(:,i));

        dataTrain = data(train,:);
        dataValid = data(valid,:);

        trainPlus = [dataTrain;dataValid]; % New Training dataset

        pool{i,1} = BaggingHomogeneous100(100,regressorName,trainPlus,1,neighbors,neurons);
        unique{i,1} = trainDataTraining(trainPlus(:,2:end),trainPlus(:,1),regressorName,neighbors,neurons);       

    end
    filenamePhase2 = strcat(resultsPathPhase2, dataset,'-', num2str(exec), '-100.mat');
    save(filenamePhase2,'pool','unique');      
end
function [errorRate, prediction] = runMINE(dataset,modelType)

load('MINE/results/configs')

fprintf('\nDATASET: %s\n', dataset);

for exec = 1 : executions

    fprintf('%s-%d\n', dataset,exec);

    %%%%% Phase 1 - Load the datasets
    filePhase1 = strcat(resultsPathPhase1, dataset, '-', num2str(exec), '-10.mat');
    load(filePhase1,'trainIndex','validIndex','testIndex','data')
    
    %%%%% Phase 2 - Load the Trained Homogeneous Ensemble and Individual Regressor
    filePhase2 = strcat(resultsPathPhase2, dataset, '-', num2str(exec), '-100.mat');
    load(filePhase2, 'pool', 'unique');
    
    sL = size(sizeL,2); % Ensemble size options

    finalBestWeightsS = zeros(folds,sL,measures); % Final weights after the GA
    finalBestWeightsW = zeros(folds,sL,measures);
    finalBestWeightsWS = zeros(folds,sL,measures);    

    techniques = 13;

    errorRate = cell(sL,techniques,folds);
    
    errorRateMeasuresDS = cell(sL,measures,folds);
    errorRateMeasuresDW = cell(sL,measures,folds);
    errorRateMeasuresDWS = cell(sL,measures,folds);
    
    prediction = cell(techniques,folds);

    % {INDIVIDUAL, DS, MINE-DS, DW, MINE-WS, DWS, MINE-DWS, MEAN, MEDIAN, ORACLE}
    errorsPerEnsemble = zeros(sL,10);
    
    predictDataTestTempMEMO = cell(folds,1);
    predictDataTrainTempMEMO = cell(folds,1);
    dataTestMEMO = cell(folds,1);
    dataTrainPlusMEMO = cell(folds,1);
    predictUniqueMEMO = cell(folds,1);
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % To improve the speed of code processing, we first run all models 
    % trained in Phase 2, storing all predictions for all training and 
    % testing patterns. It is faster in MATLAB to perform the prediction 
    % of a dataset at once than to perform the prediction for each pattern 
    % individually.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:folds
        train = logical(trainIndex(:,i));
        test = logical(testIndex(:,i));        
        valid = logical(validIndex(:,i));

        dataTrain = data(train,:);
        dataTest = data(test,:);
        dataValid = data(valid,:);
        
        dataTrainPlus = [dataTrain;dataValid];

        [rTest,~] = size(dataTest);
        [rTrain,~] = size(dataTrainPlus);

        predictDataTestTemp = zeros(rTest,100);
        predictDataTrainTemp = zeros(rTrain,100);
        
        poolTemp = pool{i,1};
        
        predictUnique = testDataTesting(dataTest(:,2:end), unique{i}, modelType);
        
        testData = dataTest(:,2:end);
        trainData = dataTrainPlus(:,2:end);
        
        parfor j = 1:100
            predictDataTestTemp(:,j) = testDataTesting(testData, poolTemp{j,1}, modelType);
            predictDataTrainTemp(:,j) = testDataTesting(trainData, poolTemp{j,1}, modelType);
        end
                
        predictDataTestTempMEMO{i,1} = predictDataTestTemp;
        predictDataTrainTempMEMO{i,1} = predictDataTrainTemp;
        dataTestMEMO{i,1} = dataTest;
        dataTrainPlusMEMO{i,1} = dataTrainPlus;
        predictUniqueMEMO{i,1} = predictUnique;
                
    end
    %%%%%%%%% FREE MEMORY %%%%%%%%%
    clear trainIndex;
    clear testIndex;
    clear validIndex;
   
    clear train;
    clear test;
    clear valid;

    clear dataTrain;
    clear dataTest;
    clear dataValid;
    
    clear predictDataTestTemp;
    clear predictDataTrainTemp;
    clear dataTest;
    clear dataTrainPlus;
    clear predictUnique;

    for i = 1:folds
           
            dataTest = dataTestMEMO{i,1};
            dataTrainPlus = dataTrainPlusMEMO{i,1};
            predictDataTest = predictDataTestTempMEMO{i,1};
            predictDataTrain = predictDataTrainTempMEMO{i,1};
            
            % Extract the measures
            matrixValuesTestTemp = extractAttValues(dataTest,dataTrainPlus,predictDataTest,predictDataTrain,1,100,K,attributes);
            matrixValuesTrainTemp = extractAttValues(dataTrainPlus,dataTrainPlus,predictDataTrain,predictDataTrain,1,100,K,attributes);
            
            predictUnique = predictUniqueMEMO{i,1};
            
            clear dataTest;
            clear dataTrainPlus;
            clear predictDataTest;
            clear predictDataTrain;

            dataTest = dataTestMEMO{i,1};
            dataTrainPlus = dataTrainPlusMEMO{i,1};

            indexSizeL = 0;
            
            for L=sizeL
                indexSizeL = indexSizeL + 1;

                predictDataTest = predictDataTestTempMEMO{i,1}(:,1:L);
                predictDataTrain = predictDataTrainTempMEMO{i,1}(:,1:L);
                
                matrixValuesTest = matrixValuesTestTemp(:,1:L,:);
                matrixValuesTrain = matrixValuesTrainTemp(:,1:L,:);
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % DS Algorithm
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                [errorRate{indexSizeL,1,i}, ~] = simpleDS(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 18);
                [errorRateMeasuresDS{indexSizeL,1,i}, ~] = simpleDS(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 5);
                [errorRateMeasuresDS{indexSizeL,2,i}, ~] = simpleDS(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 7);
                [errorRateMeasuresDS{indexSizeL,3,i}, ~] = simpleDS(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 9);
                [errorRateMeasuresDS{indexSizeL,4,i}, ~] = simpleDS(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 11);
                [errorRateMeasuresDS{indexSizeL,5,i}, ~] = simpleDS(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 13);
                [errorRateMeasuresDS{indexSizeL,6,i}, ~] = simpleDS(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 15);
                errorRateMeasuresDS{indexSizeL,7,i} = errorRate{indexSizeL,1,i};
                [errorRateMeasuresDS{indexSizeL,8,i}, ~] = simpleDS(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 19);

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%% MINE-S Algorithm
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
                FitnessHolderDS = @(X)objectiveFunction(X, dataTrainPlus, predictDataTrain, 1, matrixValuesTrain, attributes, 1);
                bestWeightsS = ga(FitnessHolderDS, cAtt, [],[],[],[],lb,ub,[], gaoptions);
                [errorRate{indexSizeL,2,i}] = process(bestWeightsS,dataTest,predictDataTest,1,matrixValuesTest,attributes,1);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % DW Algorithm
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                [errorRate{indexSizeL,3,i}, ~] = simpleDW(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 18);
                [errorRateMeasuresDW{indexSizeL,1,i}, ~] = simpleDW(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 5);
                [errorRateMeasuresDW{indexSizeL,2,i}, ~] = simpleDW(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 7);
                [errorRateMeasuresDW{indexSizeL,3,i}, ~] = simpleDW(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 9);
                [errorRateMeasuresDW{indexSizeL,4,i}, ~] = simpleDW(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 11);
                [errorRateMeasuresDW{indexSizeL,5,i}, ~] = simpleDW(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 13);
                [errorRateMeasuresDW{indexSizeL,6,i}, ~] = simpleDW(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 15);
                errorRateMeasuresDW{indexSizeL,7,i} = errorRate{indexSizeL,3,i};
                [errorRateMeasuresDW{indexSizeL,8,i}, ~] = simpleDW(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 19);

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%% MINE-W Algorithm
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
                FitnessHolderDW = @(X)objectiveFunction(X, dataTrainPlus, predictDataTrain, 1, matrixValuesTrain, attributes, 2);
                bestWeightsW = ga(FitnessHolderDW, cAtt, [],[],[],[],lb,ub,[], gaoptions);
                [errorRate{indexSizeL,4,i}] = process(bestWeightsW,dataTest,predictDataTest,1,matrixValuesTest,attributes,2);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % DWS Algorithm
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                [errorRate{indexSizeL,5,i}, ~] = simpleDWS(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 50, 18);
                [errorRateMeasuresDWS{indexSizeL,1,i}, ~] = simpleDWS(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 50, 5);
                [errorRateMeasuresDWS{indexSizeL,2,i}, ~] = simpleDWS(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 50, 7);
                [errorRateMeasuresDWS{indexSizeL,3,i}, ~] = simpleDWS(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 50, 9);
                [errorRateMeasuresDWS{indexSizeL,4,i}, ~] = simpleDWS(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 50, 11);
                [errorRateMeasuresDWS{indexSizeL,5,i}, ~] = simpleDWS(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 50, 13);
                [errorRateMeasuresDWS{indexSizeL,6,i}, ~] = simpleDWS(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 50, 15);
                errorRateMeasuresDWS{indexSizeL,7,i} = errorRate{indexSizeL,5,i};
                [errorRateMeasuresDWS{indexSizeL,8,i}, ~] = simpleDWS(dataTest, predictDataTest, 1, matrixValuesTest, attributes, 50, 19);

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%% MINE-WS Algorithm
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
                FitnessHolderDWS = @(X)objectiveFunction(X, dataTrainPlus, predictDataTrain, 1, matrixValuesTrain, attributes, 3);
                bestWeightsWS = ga(FitnessHolderDWS, cAtt, [],[],[],[],lb,ub,[], gaoptions);
                [errorRate{indexSizeL,6,i}] = process(bestWeightsWS,dataTest,predictDataTest,1,matrixValuesTest,attributes,3);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%               

                %%%%%%%%%%%%% Save all the weights
                finalBestWeightsS(i,indexSizeL,:) = bestWeightsS;
                finalBestWeightsW(i,indexSizeL,:) = bestWeightsW;
                finalBestWeightsWS(i,indexSizeL,:) = bestWeightsWS;
                
                
                % Individual Regressor
                errorRate{indexSizeL,7,i} = (predictUnique(:,1) - dataTest(:,1)).^2;

                % Bagging (Mean)
                [errorRate{indexSizeL,8,i}, ~] = baggingEvaluation(dataTest,predictDataTest,1);  

                % Median
                [errorRate{indexSizeL,9,i}, ~] = medianEvaluation(dataTest,predictDataTest,1);
                
               % Oracle
                [errorRate{indexSizeL,10,i}, ~] = simpleOracle(dataTest,predictDataTest, 1); 
                
            end
    end
    
    % [DS, MINE-S, DW, MINE-W, DWS, MINE-WS, INDIVIDUAL, MEAN, MEDIAN]
    indexSL = 0;

    for sln = sizeL 
        indexSL = indexSL + 1;

        for tec = 1 : 9
            resultsMeans = 0;
            
            for ffolds = 1 : folds
                resultsMeans = resultsMeans + mean(errorRate{indexSL,tec,ffolds});
            end
            
            resultsMeans = resultsMeans / folds;

            errorsPerEnsemble(indexSL,tec) = resultsMeans;
        end
    end
    
    fileMINE = strcat(resultsPathPhases3_4, dataset,'-',num2str(exec), '-MINE.mat');
    save(fileMINE,'finalBestWeightsS','finalBestWeightsW','finalBestWeightsWS','errorRate','errorRateMeasuresDS','errorRateMeasuresDW','errorRateMeasuresDWS');

end

end
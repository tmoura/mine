configFile = 'MINE/results/configs';

load(configFile,'datasetsUsed','modelsUsed','executions')
load(configFile,'resultsPathPhase1')

for i=1:length(datasetsUsed)
    
    filenameMATFinal = strcat(resultsPathPhase1, datasetsUsed{i}, '- FINAL -','10.mat');
    load(filenameMATFinal,'indexBetterModel');

    for j = 1:executions
        
        runGenerateEnsemble100(datasetsUsed{i},j,modelsUsed{indexBetterModel}{1}, modelsUsed{indexBetterModel}{2}, modelsUsed{indexBetterModel}{3});
    end 
end
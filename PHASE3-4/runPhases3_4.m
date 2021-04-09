configFile = 'MINE/results/configs';

load(configFile,'resultsPathPhase1')
load(configFile,'datasetsUsed')
load(configFile,'modelsUsed')

for i = 1 : length(datasetsUsed)
    
    filePhase1 = strcat(resultsPathPhase1, datasetsUsed{i}, '- FINAL -','10.mat');
    load(filePhase1,'indexBetterModel');

    runMINE(datasetsUsed{i},modelsUsed{indexBetterModel}{1});

end
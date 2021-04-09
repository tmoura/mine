function errosData = calculateErrors(pool, data, regressors)

    sRegressors = size(regressors,1);
    sData = size(data,1);
    
    predictData = zeros(sData,sRegressors);
    errosData = zeros(1,sRegressors);
    
    for nRegressor = 1 : sRegressors
        predictData(:,nRegressor) = testDataTesting(data(:,2:end), pool{nRegressor}, regressors{nRegressor}{1});
        errosData(1,nRegressor) = mean((predictData(:,nRegressor) - data(:,1)).^2);
    end
    
end
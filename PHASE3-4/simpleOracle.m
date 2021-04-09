function [errorRate, prediction] = simpleOracle(testSet,predTestSet,columnTarget)
    
    [rows,~] = size(testSet);
    
    prediction = zeros(rows,1);    
    errorRate = zeros(rows,1);     
    
    testTarget = testSet(:,columnTarget);
    
    for i=1:rows                         
        
        [~,indexMin] = min((testTarget(i,:) - predTestSet(i,:)).^2);
        
        prediction(i) = mean(predTestSet(i,indexMin));      
        errorRate(i) = (testTarget(i) - prediction(i))^2;
        
    end
end
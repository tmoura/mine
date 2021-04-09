function [errorsRate, prediction] = baggingEvaluation(testSet,predTestSet,columnTarget)
        
    prediction = mean(predTestSet,2);
    errorsRate = (testSet(:,columnTarget) - prediction).^2;

end
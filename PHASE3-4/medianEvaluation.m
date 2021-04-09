function [errorsRate, prediction] = medianEvaluation(testSet,predTestSet,columnTarget)

    prediction = median(predTestSet,2);
    errorsRate = (testSet(:,columnTarget) - prediction).^2;

end
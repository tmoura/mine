function [fitness] = objectiveFunction(weights,optSet,predOptSet,columnTarget,matrixValues,att, op)                       

    [errorRate] = process(weights,optSet,predOptSet,columnTarget,matrixValues,att,op);

    fitness = mean(errorRate);  
end
function [matrixValues] = extractAttValues(testSet,valSet,predTestSet,predValSet,columnTarget,poolSize,K,att)

    [rows,~] = size(testSet);

    [~,cAtt] = size(att);
    
    %%%%%%%%%%%%%%%%%%%%%%%
    % (number of attributes, pool size, number of patterns being evaluated)
    %%%%%%%%%%%%%%%%%%%%%%%
    matrixValues = zeros(cAtt,poolSize,rows);
    
    valTargets = valSet(:,columnTarget);
    valSet(:,columnTarget) = [];        
    
    for i=1:rows

        point = testSet(i,:);
        point(:,columnTarget) = [];

        [distance,index] = knn(valSet, point, K, 1);   % Nearest Neighbours
        pesos = (1./distance) / sum((1./distance));        

        neighbor = predValSet(index,:);
                    
        for k = 1 : cAtt
            matrixValues(k,:,i) = attribute(valTargets(index,:), neighbor, valSet(index,:), point, predTestSet(i,:), pesos, att(k));
        end
    end
end
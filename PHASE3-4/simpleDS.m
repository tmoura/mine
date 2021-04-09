function [errorsRate, prediction] = simpleDS(testSet, predTestSet, columnTarget, matrixValues, att, measure)

    pos = (att == measure);
    
    testTarget = testSet(:,columnTarget);

    C = permute(matrixValues(pos,:,:),[1 3 2]);
    C = reshape(C,[],size(matrixValues(pos,:,:),2),1);
    
    [~,iMin] = min(C,[],2);
    
    I = (1 : size(predTestSet, 1)) .';
    J = reshape(iMin, [], 1);
    k = sub2ind(size(predTestSet), I, J);
    
    prediction = predTestSet(k);   
    
    errorsRate = (testTarget - prediction).^2;
end
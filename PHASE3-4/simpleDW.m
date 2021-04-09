function [errorsRate, prediction] = simpleDW(testSet, predTestSet, columnTarget, matrixValues, att, measure)

    pos = (att == measure);
    
    testTarget = testSet(:,columnTarget);
    
    C = permute(matrixValues(pos,:,:),[1 3 2]);
    C = reshape(C,[],size(matrixValues(pos,:,:),2),1);
 
    numerator = 1./C;
    [v,~,~] = ind2sub(size(numerator),find(numerator == Inf)); % Find the Infs

    if ~isempty(v)
        tempNumerator = numerator(v,:);    
        tempNumerator(tempNumerator ~= Inf) = 0;
        tempNumerator(tempNumerator == Inf) = 1;
        numerator(v,:) = tempNumerator;
    end
    
    denominador = sum(numerator,2);

    weightsNEW = numerator./denominador;

    prediction = sum(weightsNEW .* predTestSet,2);
    errorsRate = (testTarget - prediction).^2;
    
end
function [errorsRate, prediction] = simpleDWS(testSet, predTestSet, columnTarget, matrixValues, att, percentual, measure)

    pos = (att == measure);
    
    testTarget = testSet(:,columnTarget);
    
    C = permute(matrixValues(pos,:,:),[1 3 2]);
    C = reshape(C,[],size(matrixValues(pos,:,:),2),1);

    Emax = max(C,[],2);
    Emin = min(C,[],2);
    E = (Emax - Emin) * (percentual/100);

    iPred = C <= E;

    iPred(sum(iPred,2) == 0,:) = 1;
    
    C(C == 0) = NaN;
    
    numerator = C .* iPred;
    numerator = 1./numerator;
    numerator(numerator == Inf) = 0;  % Change Inf to 0
    
    [v,~,~] = ind2sub(size(numerator),find(isnan(numerator))); % Find the indexes with NaN

    if ~isempty(v)
        tempNumerator = numerator(v,:);    
        tempNumerator(~isnan(tempNumerator)) = 0;
        tempNumerator(isnan(tempNumerator)) = 1;
        numerator(v,:) = tempNumerator;
    end
        
    denominator = sum(numerator,2);

    weightsNEW = numerator ./ denominator;

    prediction = sum(weightsNEW .* predTestSet,2);
    errorsRate = (testTarget - prediction).^2;
    
end
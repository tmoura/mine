%%%%%%%%%%%%%%%%%%%%%%
%
% Function that calculates the fitness of the genetic algorithm
%
%%%%%%%%%%%%%%%%%%%%%%
function [result] = process(weights,optSet,predOptSet,columnTarget,matrixValues,att,option)
        
    optSet = optSet(:,columnTarget);
        
    if option == 1  %%% MINE-S

        tempResults = DSProcess(weights, matrixValues, predOptSet);

    elseif option == 2  %%% MINE-W

        tempResults = DWProcess(weights, matrixValues, predOptSet);

    elseif option == 3  %%% MINE-WS

        tempResults = DWSProcess(weights, matrixValues, predOptSet);
    end
        
    result = (optSet - tempResults).^2;
end

function rresult = DSProcess(weights, matrixValues, predOptSet)
       
        temp = bsxfun(@times,weights',matrixValues(:,:,:));
        resultSum = sum(temp);
        
        [~,iMin] = min(resultSum);
        
        idx = bsxfun(@eq, cumsum(ones(size(predOptSet)), 2), iMin(:));
        rresult = sum(predOptSet.*idx, 2);
       
end

function rresult = DWProcess(weights, matrixValues, predOptSet)
        
        values = sum(weights' .* matrixValues(:,:,:));

        [~,~,v] = ind2sub(size(values),find(values == 0));

        if ~isempty(v)

            values(:,:,v) = matrixValues(3,:,v);
        
        end

        numerador = 1./(values);
        denominador = sum(1./(values));

        weightsNEW = numerador./denominador;
        
        C = permute(weightsNEW,[1 3 2]);
        C = reshape(C,[],size(weightsNEW,2),1);
        
        rresult = sum(C .* predOptSet,2);
end

function rresult = DWSProcess(weights, matrixValues, predOptSet)
         
        values = sum(weights' .* matrixValues(:,:,:));

        [~,~,v] = ind2sub(size(values),find(values == 0));

        if ~isempty(v)

            values(:,:,v) = matrixValues(3,:,v);

        end

        Emax = max(values);
        Emin = min(values);

        E = (Emax - Emin) * (50/100);

        iPred = values <= E; % Matrix with the values 0

        iPred(:,:,sum(iPred) == 0) = 1; % Change the indexes [0 0 0 0...] for [1 1 1 1...]

        numerador = values(:,:,:) .* iPred(:,:,:);
        numerador = 1./numerador;
        numerador(numerador == Inf) = 0;  % Change the Infs for 0

        denominador = sum(numerador);

        weightsNEW = numerador ./ denominador;

        C = permute(weightsNEW,[1 3 2]);
        C = reshape(C,[],size(weightsNEW,2),1);

        rresult = sum(C .* predOptSet,2);       
 
end
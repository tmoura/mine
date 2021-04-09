%%%%%%%%%%%%%%%%%%
%
% List of measures that can be taken from the generated region of 
% competence for a given test pattern and regressor performance in the region
%
% Parameters:
%
% valTagets - Vector with real values of neighbors
% predValTargets - Matrix with predicted values for neighbors
% point - Vector with the characteristics of the point being tested
% predPoint - Vector with predictions of the point being tested
% dist - Weighted distance vector (closer, higher value)
% att - Scalar representing the characteristic to be calculated
%
% ===> ATTENTION <===
%
% MANY OTHER MEASURES WERE IMPLEMENTED IN AN ATTEMPT TO IMPROVE THE PERFORMANCE 
% OF THE FRAMEWORK, BUT ONLY THE MEASURES [5 7 9 11 13 15 18 19] WERE USED, 
% AS DESCRIBED IN THE PUBLISHED FRAMEWORK ARTICLE
%
% AS FUTURE WORK, SOME OTHER METRICS CAN BE ADDED AND USED TO CHECK PERFORMANCE.
%%%%%%%%%%%%%%%%%%
function [result] = attribute(valTargets, predValTargets, valSet, point, predPoint, dist, att)

    switch att
        case 1
            result = MAPE(valTargets, predValTargets);
        case 2
            result = votes(valTargets, predValTargets, 0, dist);
        case 3
            result = votes(valTargets, predValTargets, 1, dist);
        case 4
            result = pesosDistance(valTargets, predValTargets, dist);
        case 5
            result = variancia(predValTargets, valTargets);
        case 6
            result = erroTreinamento(data, pool);
        case 7
            result = erroTotalAbsoluto(valTargets, predValTargets, dist);
        case 8
            result = erroTotalMSE(valTargets, predValTargets);
        case 9
            result = erroTotalMSEDistancia(valTargets, predValTargets, dist);
        case 10
            result = erroMinimo(valTargets, predValTargets);
        case 11
            result = erroMinimoDistancia(valTargets, predValTargets, dist);
        case 12
            result = erroMaximo(valTargets, predValTargets);
        case 13
            result = erroMaximoDistancia(valTargets, predValTargets, dist);
        case 15
            result = somatorioDiferencasDistanciaReal(predPoint, valTargets, dist);
        case 16
            result = somatorioDiferencasDistanciaPred(predPoint, predValTargets, dist);
        case 17
            result = predPoint;
        case 18
            result = erroTotalRMSEDistancia(valTargets, predValTargets, dist);
        case 19
            result = erroVizinhoMaisProximo(valTargets, predValTargets);
        case 20
            result = erroTree(valTargets, valSet, point, predPoint);
        case 21
            result = erroLinear(valTargets, valSet, point, predPoint);
    end

end

function [result] = erroTotalAbsoluto(valTargets, predValTargets, distancias)
    
    result = sum(abs(valTargets - predValTargets).* distancias);
end

function [result] = erroTotalMSE(valTargets, predValTargets)

    result = sum((valTargets - predValTargets).^2);

end

function [result] = erroTotalMSEDistancia(valTargets, predValTargets, distancias)
    
    result = sum(((valTargets - predValTargets).^2) .* distancias);

end

function [result] = MAPE(valTargets, predValTargets)
    
    result = mean((abs((valTargets - predValTargets)./valTargets * 100)));

end

function [result] = votes(valTargets, predValTargets, peso, distancias)
    
    minimos = abs(valTargets - predValTargets);
    
    [rMin, ~] = size(minimos);
    
    result = zeros(1,4);
    
    for k = 1 : rMin
         [~,indexMin] = min(minimos(k,:));
        
         if (peso == 1)
            result(:,indexMin) = result(:,indexMin) + (1 * distancias(k));
         else
             result(:,indexMin) = result(:,indexMin) + 1;
         end
    end
end

function [result] = pesosDistance(valTargets, predValTargets, distancia)
    
    minimos = abs(valTargets - predValTargets);
    
    [rMin, ~] = size(minimos);
    
    result = zeros(1,4);
    
    for k = 1 : rMin
         [~,indexMin] = min(minimos(k,:));
         
         result(:,indexMin) = result(:,indexMin) + distancia(k);
    end
end

function [result] = variancia(predValTargets, valTargets)

    result = var(predValTargets); 

end

function [result] = erroTreinamento(data, pool)

    [rPool,~] = size(pool);
    
    result = zeros(1,rPool);

    for i = 1:rPool
        regressor = pool{i};

        if (i == 1) 
            type = 'tree'; end
        if (i == 2) 
            type = 'svr'; end
        if ((i == 3) || (i == 4)) 
            type = 'mlp'; end

        predictData = testRegressor(data(:,2:end), regressor, type);
        result(i) = sum((predictData - data(:,1)).^2);
    end
    
end

function [result] = erroMinimo(valTargets, predValTargets)

    result = min((valTargets - predValTargets).^2);

end

function [result] = erroMinimoDistancia(valTargets, predValTargets, distancias)

    result = min(((valTargets - predValTargets).^2) .* distancias);

end

function [result] = erroMaximo(valTargets, predValTargets)

    result = max((valTargets - predValTargets).^2);

end


function [result] = erroMaximoDistancia(valTargets, predValTargets, distancias)

    result = max(((valTargets - predValTargets).^2) .* distancias);

end

function [result] = somatorioDiferencasDistanciaReal(predPoint, valTargets, distancias)

    result = sum(((predPoint - valTargets).^2) .* distancias);

end

function [result] = somatorioDiferencasDistanciaPred(predPoint, predValTargets, distancias)

    result = sum((predPoint - predValTargets).^2 .* distancias);

end

function [result] = erroTotalRMSEDistancia(valTargets, predValTargets, distancias)
    
    result = sqrt(sum(((valTargets - predValTargets).^2) .* distancias));

end

function [result] = erroVizinhoMaisProximo(valTargets, predValTargets)
    
    result = (valTargets(1,:) - predValTargets(1,:)).^2;

end

function [result] = erroTree(valTargets, valSet,point, predPoint)

    tree = trainRegressor(valSet, valTargets, 'tree', 0, 0);
    predict =  testRegressor(point, tree, 'tree');
    
    result = (predict - predPoint).^2;
end

function [result] = erroLinear(valTargets, valSet, point, predPoint)

    tree = trainRegressor(valSet, valTargets, 'linear', 0, 0);
    predict =  testRegressor(point, tree, 'linear');
    
    result = (predict - predPoint).^2;
end
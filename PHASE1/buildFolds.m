%%%%%%%%%%
%
% FUNCTION: buildFolds
%
% OBJECTIVE: The purpose of this function is to build the k-folds.
%
% INPUT:
%
% - allLabels: All the laabels
% - folds: Number oof the folds
% - Tr, Va, Te: The sizes of the folds
%
% OUTPUT:
%
% - testIndex: Test indexes
% - validIndex: Validation indexes
% - trainIndex: Train indexes
%
% Example:
%
% >> buildFolds(labels, 10, 7, 2, 1);
%
%%%%%%%%%%
function [trainIndex, validIndex, testIndex] = buildFolds(allLabels, folds, Tr, Va, Te)
  
    if ((Tr + Te + Va) ~= folds)
        error('The sum of the number of sets must be equal to %d', folds);
    end

    N = length(allLabels);  

    indices = crossvalind('Kfold',N,folds);

    testIndex = [];
    validIndex = [];
    trainIndex = [];
    
    conjuntos = cell(1,folds);
        
    for i=1:folds
        c = [];    
        for x = i : folds
            c = [c,x];
        end
        for y = 1 : i-1
            c = [c,y];
        end
        conjuntos{i} = c;
    end
    
    for i = 1 : folds
        
        dadosFolds = conjuntos{i};
        train = [];
        test = [];
        valid = [];
        
        %%% Train %%%
        
            for x = 1 : Tr
                if(isempty(train))
                    train = (indices == dadosFolds(x));
                else
                    train = train + (indices == dadosFolds(x));
                end
            end

        %%% Test %%%
        
            for y = x + 1 : x + Te
                if(isempty(test))
                    test = (indices == dadosFolds(y));
                else
                    test = test + (indices == dadosFolds(y));
                end
            end

        %%% Validation %%%
    
            for z = y + 1 : y + Va
                if(isempty(valid))
                    valid = (indices == dadosFolds(z));
                else
                    valid = valid + (indices == dadosFolds(z));
                end
            end

        
        testIndex = [testIndex,test];
        validIndex = [validIndex,valid];
        trainIndex = [trainIndex,train];
        
    end
    
end
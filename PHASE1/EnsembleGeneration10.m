%%%%%%%%%%%%
%
% T - size of the set
% data - dataset
% columnY - column with the observed value
%
%%%%%%%%%%%%
function set = EnsembleGeneration10(T, data, columnY)

    set = cell(T,1);  % set of regressors

    labels = data(:,columnY);
    data(:,columnY) = [];
    
    load('configs')
    
    for i = 1:T
        set{i} = trainDataTraining(data, labels, modelsUsed{i}{1}, modelsUsed{i}{2}, modelsUsed{i}{3});     
    end
    
end
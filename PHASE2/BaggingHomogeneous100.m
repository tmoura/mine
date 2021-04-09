%%%%%%%%%%
%
% FUNCTION: BaggingHomogeneous100
%
% OBJECTIVE: The purpose of this function is to receive a training set, 
% generate S boostrap samples to train S models and store them in a Pool.
%
% INPUT:
%
% - S: Ensemble Size
% - baseLearn: Learning Algoritm
% - data: Training Data
% - columnClass: Column with the label
%
% OUTPUT:
%
% - pool: The homogeneous ensemble
%
%%%%%%%%%%
function pool = BaggingHomogeneous100(S, baseLearn, data, columnClass, K, neurons)

    [~,bootsam] = bootstrp(S,[],data);
    pool = cell(S,1);  % empty pool with size S
   
    for i=1:S
        x = data(bootsam(:,i),:);   % get the data to training
        y = x(:,columnClass);       % get the label column
        x(:,columnClass) = []; 

        pool{i} = trainDataTraining(x, y, baseLearn, K, neurons);
    end

end
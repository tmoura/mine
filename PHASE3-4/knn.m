%%%%%%%%%%
%
% FUNCTION: knn
%
% The purpose of this function is to return the nearest neighbors of a 
% given point in the feature space. Euclidean distance is used.
%
% INPUT:
%
% - dataSet: Other instances
% - point: Instance being evaluated
% - K: Number of nearest neighbors
%
% OUTPU:
%
% - index: nearest neighbors indexes
% - value: distance values
%
%%%%%%%%%%
function [value,index] = knn(dataSet, point, K, columnClass)

    dist = sqrt(sum((dataSet - point).^2,2));
    
    [value,index] = sort(dist);
        
    index = index(value ~= 0);
    value = value(value ~= 0);
    
    index = index(1:K,1);
    value = value(1:K,1);
end
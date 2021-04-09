classdef KNNRegressor
         properties  %(GetAccess = private)        
            neighborhood = 0;
            labels = [];
            data = [];
         end 
         methods
             %constructor
             function obj = KNNRegressor(n,l,d)
                obj.neighborhood = n;
                obj.labels = l;
                obj.data = d;
             end
             
             function [result] = calculate(points,obj)
                [rows,~] = size(obj.data);
                [rowsPoints,~] = size(points);
                
                result = zeros(rowsPoints,1);
                
                if (rows < obj.neighborhood)
                    error('You specified more neighbors than the number of existing points.')
                end
                
                for j = 1 : rowsPoints

                    point = points(j,:);

                    dist = sqrt(sum((obj.data - point).^2,2));

                    [~,index] = sort(dist);

                    values = obj.labels(index(1:obj.neighborhood,1),1);

                    result(j) = mean(values);
                end
             end
         end 
end
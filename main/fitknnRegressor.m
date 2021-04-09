function [regressor] = fitknnRegressor(data, labels, K)

    regressor = KNNRegressor(K,labels,data);

end

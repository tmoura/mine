function [results] = testDataTesting(data, regressor, type)
    
    if strcmp(type,'cart')       
        results = test_cart(data, regressor);

    elseif strcmp(type,'linear')       
        results = test_linear(data, regressor);
        
    elseif strcmp(type,'knn')       
        results = test_knn(data, regressor);

    elseif strcmp(type,'mlp')       
        results = test_mlp(data, regressor);
        
    elseif strcmp(type,'svr-linear')       
        results = test_svr(data, regressor);

    elseif strcmp(type,'svr-rbf')       
        results = test_svr(data, regressor);

    elseif strcmp(type,'svr-poly1')       
        results = test_svr(data, regressor);

    elseif strcmp(type,'svr-poly3')       
        results = test_svr(data, regressor);

    elseif strcmp(type,'rbf')       
        results = test_rbf(data, regressor);
    end

end

function [results] = test_svr(data, regressor)

    results = predict(regressor,data);
    
end

function [results] = test_cart(data, regressor)

    results = predict(regressor,data);
    
end

function [results] = test_linear(data, regressor)

    results = predict(regressor,data);
    
end

function [results] = test_knn(data, regressor)

    results = calculate(data,regressor);
    
end

function [results] = test_mlp(data, regressor)

    results = regressor(data');
    results = results';

end

function [results] = test_rbf(data, regressor)

    results = regressor(data');
    results = results';

end
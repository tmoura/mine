function regressor = trainDataTraining(data, labels, type, K, neurons)
    
    if strcmp(type,'cart')       
        regressor = train_cart(data, labels);

    elseif strcmp(type,'linear')       
        regressor = train_linear(data, labels);
        
    elseif strcmp(type,'knn')       
        regressor = train_knn(data, labels, K);

    elseif strcmp(type,'mlp')       
        regressor = train_mlp(data, labels, neurons);
        
    elseif strcmp(type,'svr-linear')       
        regressor = train_svrLINEAR(data, labels);

    elseif strcmp(type,'svr-rbf')       
        regressor = train_svrRBF(data, labels);

    elseif strcmp(type,'svr-poly1')       
        regressor = train_svrPOLY1(data, labels);

    elseif strcmp(type,'svr-poly3')       
        regressor = train_svrPOLY3(data, labels);

    elseif strcmp(type,'rbf')       
        regressor = train_rbf(data, labels, neurons);
    end

end

function mlp = train_mlp(data, labels, neurons)

    mlp = feedforwardnet(neurons);
    
    mlp.trainParam.showWindow = false;
        
    [mlp, ~] = train(mlp,data',labels');
    
end

function svr = train_svrLINEAR(data, labels)
    
    svr = fitrsvm(data,labels);
end

function svr = train_svrRBF(data, labels)
    
    svr = fitrsvm(data,labels,'KernelFunction','rbf');
    
end

function svr = train_svrPOLY1(data, labels)
    
    svr = fitrsvm(data,labels,'KernelFunction','polynomial','PolynomialOrder',1);
end

function svr = train_svrPOLY3(data, labels)

    svr = fitrsvm(data,labels,'KernelFunction','polynomial','PolynomialOrder',3);
end

function rbf = train_rbf(data, labels, neurons)
   
    P = data';	
    T = labels';	
    goal = 0.0;
    spread = 1.0;
    MN = neurons;
    DF = 25;

    rbf = newrb(P,T,goal,spread,MN,DF);
    
end

function tree = train_cart(data, labels)

    tree = fitrtree(data,labels);
    
end

function lm = train_linear(data, labels)

    lm = fitlm(data,labels);

end

function knn = train_knn(data, labels, K)
  
    knn = fitknnRegressor(data, labels, K);
   
end
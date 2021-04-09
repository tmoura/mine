datasetsUsed = {
    'abalone',...
    'airfoil_self_noise',...
    'bank32nh',...
    'bank8FM',...
    'wiscoinBreastCancer',...
    'ccpp',...
    'cpu_act',...
    'cpu_small',...
    'concrete',...
    'delta_ailerons',...
    'delta_elevators',...
    'housing',...
    'kin8nm',...
    'machine',...
    'puma32H',...
    'puma8NH',...
    'stocks',...
    'triazines',...
    'wineq-red',...
    'wineq-white'};

% To test just 1 dataset, uncomment the next line and change the dataset
% name

% datasetsUsed = {'concrete'};

%%%%%%%%%%%%
%
% ModelsUsed: {model_name, number_of_neighbors, number_of_neurons}
%
%%%%%%%%%%%%
modelsUsed = {{'cart',0,0}
              {'linear',0,0}
              {'mlp',0,10}
              {'mlp',0,[5,10]}
              {'svr-rbf',0,0}
              {'svr-poly1',0,0}
              {'svr-poly3',0,0}
              {'rbf',0,10}
              {'knn',3,0}
              {'knn',5,0}};


%%%%%%%%%%%%
% Number of executions
%%%%%%%%%%%%
executions = 20;

%%%%%%%%%%%%
% Number of folds
%%%%%%%%%%%%
folds = 10;

%%%%%%%%%%%%
% Region of Competence Size
%%%%%%%%%%%%
K = 10;

%%%%%%%%%%%%
% Ensemble sizes used
%%%%%%%%%%%%
sizeL = [5,10,15,20,30,40,50,60,70,80,90,100];

%%%%%%%%%%%%
% Attributes to be extracted from region of competence
%%%%%%%%%%%%
attributes = [5 7 9 11 13 15 18 19];
measures = length(attributes);

%%%%%%%%%%%%%%%%%%%%%
% GA Configurations
%%%%%%%%%%%%%%%%%%%%%
[~, cAtt] = size(attributes);

ip1 = zeros(cAtt,cAtt);
ip2 = diag(ones(1,cAtt));
ip=[ones(1,cAtt);ip2];

lb = zeros(1,cAtt);
ub = ones(1,cAtt);

gaoptions = optimoptions('ga','Display', 'off', ...
      'PopulationSize', 80, 'PopInitRange', [0;1], 'FitnessLimit', 0, 'StallGenLimit', 40, ...
      'InitialPopulation',ip,'EliteCount',cAtt,'UseParallel', true);
%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%
% Path to save the Results Phase to Phase
%%%%%%%%%%%%
resultsPathPhase1 = 'MINE/results/PHASE1/';
resultsPathPhase2 = 'MINE/results/PHASE2/';
resultsPathPhases3_4 = 'MINE/results/PHASE3_4/';

save('MINE/results/configs')
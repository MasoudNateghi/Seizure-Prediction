clear; close all; clc;
load("data\s5_data.mat")
%% correlation 
X_AllCorr = [XT_corr; XTest_corr];
nT = size(XT_corr, 1);
%% find maximum threshold
% Gthr = inf;
% for i = 1:size(X_AllCorr, 1)
%     A = wuGraph(X_AllCorr(i, :));
%     Tree = UndirectedMaximumSpanningTree(A);
%     thr = findMaxThreshold(Tree, A);
%     if thr < Gthr 
%         Gthr = thr;
%         "Update! Gthr = " + num2str(Gthr) + " i = " + num2str(i)
%     end
% end
%% test
i = 2;
B = wuGraph(X_AllCorr(i, :));
Tree = UndirectedMaximumSpanningTree(B);
Gthr = findMaxThreshold(Tree, B);
B = abs(X_AllCorr(i, :)) >= Gthr;
A = binGraph(B);
plotGraph(Tree)
plotGraph(A)
% [lambda,efficiency,ecc,radius,diameter] = charpath(distance_bin(B))
%% extract features from binary graph
X_Eglob = zeros(size(X_AllCorr, 1), 1);
X_Eloc = zeros(size(X_AllCorr, 1), 18);
X_deg = zeros(size(X_AllCorr, 1), 18);
X_between = zeros(size(X_AllCorr, 1), 18);
X_close = zeros(size(X_AllCorr, 1), 18);
X_pageRank = zeros(size(X_AllCorr, 1), 18);
X_eigenVector = zeros(size(X_AllCorr, 1), 18);
X_assort = zeros(size(X_AllCorr, 1), 1);
X_den = zeros(size(X_AllCorr, 1), 2);
X_flow = zeros(size(X_AllCorr, 1), 19);
X_subcen = zeros(size(X_AllCorr, 1), 18);
X_tri = zeros(size(X_AllCorr, 1), 1);
X_clust = zeros(size(X_AllCorr, 1), 18);
X_charpath = zeros(size(X_AllCorr, 1), 21);
for i = 1:size(X_AllCorr, 1)
    i %#ok<NOPTS> 
    B = wuGraph(X_AllCorr(i, :));
    Tree = UndirectedMaximumSpanningTree(B);
    Gthr = findMaxThreshold(Tree, B);
    B = abs(X_AllCorr(i, :)) >= Gthr;
    A = binGraph(B);
    X_Eglob(i) = efficiency_bin(A); % Global Efficiency
    X_Eloc(i, :) = efficiency_bin(A,1)'; % Local Efficiency
    X_deg(i, :) = degrees_und(A); % Degree of Nodes
    X_between(i, :) = centrality(graph(A), "betweenness"); % Betweenness
    X_close(i, :) = centrality(graph(A), "closeness"); % Closeness
    X_pageRank(i, :) = centrality(graph(A), "pagerank"); % PageRank
    X_eigenVector(i, :) = centrality(graph(A), "eigenvector"); % EigenVector
    X_assort(i) = assortativity_bin(A, 0); % Assortativity
    [X_den(i, 1), X_den(i, 2)] = density_und(A); % Density and Number of edges
    [X_flow(i, 1:18), X_flow(i, 19)] = flow_coef_bd(A); % Node-wise flow coefficients
    X_subcen(i, :) = subgraph_centrality(A); % Subgraph centrality of a network
    X_tri(i) = transitivity_bu(A); % Transitivity
    X_clust(i, :) = clustering_coef_bu(A)'; % Clustering coefficient
    [lambda,~,ecc,radius,diameter] = charpath(distance_bin(A)); % Lambda\ ecc\ radius\ diameter
    X_charpath(i, :) = [lambda, ecc', radius, diameter];
end
%%
% normalize features
XG = [X_Eglob, X_Eloc, X_deg, X_between, X_close, X_pageRank, X_eigenVector, X_assort, X_den, X_flow, X_subcen, X_tri, X_clust, X_charpath]';
[XGT,PS] = mapstd(XG(:, 1:nT));
XGTest = mapstd('apply',XG(:, nT+1:end), PS);
%% regression with 5-Fold cross validation
% yTest = labelTest;
[B,FitInfo] = lasso(XGT',yTrain,'CV',5);
idxLambda1SE = FitInfo.Index1SE;
coef = B(:,idxLambda1SE);
coef0 = FitInfo.Intercept(idxLambda1SE);
% plot MSE for different lambda values
lassoPlot(B,FitInfo,'PlotType','CV');
legend('show') % Show legend
% AUC and confusion matrix for train data
yhatTrain = XGT'*coef + coef0;
[X,Y,T,AUC,OPTROCPT] = perfcurve(yTrain,yhatTrain,1); 
% plot(X, Y); xlabel('fpr'); ylabel('tpr'); title('AUC for train data (corr)')
fprintf('AUC for train data = %f\n', AUC)
best_thr_index = find(X == OPTROCPT(1) & Y == OPTROCPT(2));
best_thr = T(best_thr_index); 
yTrain_predict = zeros(size(yTrain));
yTrain_predict(yhatTrain >= best_thr) = 1;
figure;
str = "Train Data";
sgtitle(str, "Interpreter","latex")
cm = confusionchart(yTrain, yTrain_predict);
cm.Normalization = 'row-normalized';
% figure;
% cm = confusionchart(yTrain, yTrain_predict);  %#ok<NASGU> 
% sgtitle(str, "Interpreter","latex")
% AUC and confusion matrix for test data
yhatTest = XGTest'*coef + coef0;
yTest_predict = zeros(size(yTest));
yTest_predict(yhatTest >= best_thr) = 1;
figure;
cm = confusionchart(yTest, yTest_predict);
str = "Test Data";
sgtitle(str, "Interpreter","latex")
cm.Normalization = 'row-normalized';
% figure;
% cm = confusionchart(yTest, yTest_predict); 
% sgtitle(str, "Interpreter", "latex")

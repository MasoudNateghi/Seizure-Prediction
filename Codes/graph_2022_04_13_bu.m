clear; close all; clc;
load("data\s5_data.mat")
%% correlation 
% for p = linspace(0.1, 0.9, 9)
p = 0.6;
%%
X_AllCorr = [XT_corr; XTest_corr];
X_AllPlv = [XT_plv; XTest_plv];
nT = size(XT_corr, 1);
% thresholding
corrMax = max(abs(X_AllCorr(:)));
Gthr = p*corrMax;
X_AllCorrBin = abs(X_AllCorr) >= Gthr;
A = binGraph(X_AllCorrBin(2, :));
% plotGraph(A)
%%
% yTest = labelTest;
% test
% A = binGraph(X_AllCorrBin(1, :));
% [C_tri]=transitivity_bu(A)
% extract features from binary graph
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
for i = 1:size(X_AllCorr, 1)
    i %#ok<NOPTS> 
    A = binGraph(X_AllCorrBin(i, :));
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
end
% normalize features
% XG = [X_Eglob, X_Eloc, X_deg, X_between, X_close, X_pageRank, X_eigenVector, X_assort, X_den, X_flow, X_subcen, X_tri]';
XG = [X_Eglob, X_Eloc, X_deg, X_between, X_close, X_pageRank, X_eigenVector, X_assort, X_den, X_flow, X_subcen, X_tri, X_AllCorr]';
% XG = [X_Eglob, X_Eloc, X_deg, X_between, X_close, X_pageRank, X_eigenVector, X_assort, X_den, X_flow, X_subcen, X_tri, X_AllCorr, X_AllPlv]'; %
[XGT,PS] = mapstd(XG(:, 1:nT));
XGTest = mapstd('apply',XG(:, nT+1:end), PS);
XGT = XGT';
XGTest = XGTest';
%% lasso
% regression with 5-Fold cross validation
clear; close all; clc;
load("data\s5_bu_graph_features.mat")
[B,FitInfo] = lasso(XGT,yTrain,'CV',5);
idxLambda1SE = FitInfo.Index1SE;
coef = B(:,idxLambda1SE);
coef0 = FitInfo.Intercept(idxLambda1SE);
% plot MSE for different lambda values
% lassoPlot(B,FitInfo,'PlotType','CV');
% legend('show') % Show legend
% AUC and confusion matrix for train data
yhatTrain = XGT*coef + coef0;
[X,Y,T,AUC,OPTROCPT] = perfcurve(yTrain,yhatTrain,1); 
% plot(X, Y); xlabel('fpr'); ylabel('tpr'); title('AUC for train data (corr)')
fprintf('AUC for train data = %f\n', AUC)
best_thr_index = find(X == OPTROCPT(1) & Y == OPTROCPT(2));
best_thr = T(best_thr_index); 
% [best_thr, acc] = find_best_thr(yTrain, yhatTrain, T);
% yTrain_predict = zeros(size(yTrain));
% yTrain_predict(yhatTrain >= best_thr) = 1;
% figure;
% str = "Train Data";
% sgtitle(str, "Interpreter","latex")
% cm = confusionchart(yTrain, yTrain_predict);
% cm.Normalization = 'row-normalized';
% figure;
% cm = confusionchart(yTrain, yTrain_predict);  %#ok<NASGU> 
% sgtitle(str, "Interpreter","latex")
% AUC and confusion matrix for test data
yhatTest = XGTest*coef + coef0;
yTest_predict = zeros(size(yTest));
yTest_predict(yhatTest >= best_thr) = 1;
% figure;
TP = sum(yTest_predict == 1 & yTest == 1);
TN = sum(yTest_predict == 0 & yTest == 0);
FN = sum(yTest_predict == 0 & yTest == 1);
FP = sum(yTest_predict == 1 & yTest == 0);
best_acc = (TP + TN) / (TP + TN + FP + FN)
best_sens = TP / (TP + FN)
best_spec = TN / (TN + FP)
% cm = confusionchart(yTest, yTest_predict);
% str = "Test Data";
% sgtitle(str, "Interpreter","latex")
% cm.Normalization = 'row-normalized';
% figure;
% cm = confusionchart(yTest, yTest_predict); 
% sgtitle(str, "Interpreter", "latex")
% end
%% svm
best_features = find(coef ~= 0);
XGTR = XGT(:, best_features); % reduced
XGTestR = XGTest(:, best_features);
%% feature selection

c = cvpartition(yTrain,'k',5);


opts = statset('display','iter');
classf = @(XGT, yTrain, XGTest, yTest)...
    sum(predict(fitcsvm(XGT, yTrain,'KernelFunction','rbf'), XGTest) ~= yTest);

[fs, history] = sequentialfs(classf, XGT, yTrain, 'cv', c, 'options', opts,'nfeatures',35);
%%
SVMModel = fitcsvm(XGTR, yTrain, "KernelFunction", "rbf", "OptimizeHyperparameters","auto", ...
    "HyperparameterOptimizationOptions",struct("AcquisitionFunctionName", 'expected-improvement-plus','ShowPlots',true));

yTrain_predict = predict(SVMModel, XGTR);
figure;
str = "Train Data with thr = " + num2str(p);
sgtitle(str, "Interpreter","latex")
cm = confusionchart(yTrain, yTrain_predict);
cm.Normalization = 'row-normalized';
yTest_predict = predict(SVMModel, XGTestR);
figure;
str = "Test Data with thr = " + num2str(p);
sgtitle(str, "Interpreter","latex")
cm = confusionchart(yTest, yTest_predict);
cm.Normalization = 'row-normalized';




















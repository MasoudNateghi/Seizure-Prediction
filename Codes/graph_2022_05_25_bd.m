%% load data
clear; close all; clc;
load("data\s1_gc.mat")
%% threshoding
nT = size(XT_gc, 3);
X_All = cat(3, XT_gc, XTest_gc);
% remove diagonal entries
for i = 1:size(X_All, 3)
    X_All(:, :, i) = X_All(:, :, i) - diag(diag(X_All(:, :, i)));
end
% find minimum maximum weight in graphs
maxel = inf;
for i = 1:size(X_All, 3)
    if max(X_All(:, :, i), [], "all") < maxel
        maxel = max(X_All(:, :, i), [], "all");
    end
end
p = 0.6;
Gthr = p * maxel;
X_AllBin = abs(X_All) >= Gthr;
X_AllSparse = X_AllBin .* X_All;
%% test
% i = 100;
% A = X_AllBin(:, :, i);
% M=link_communities(A)
%% extract features from binary graph
X_assort = zeros(size(X_All, 3), 4);
X_BC = zeros(size(X_All, 3), 18);
X_degree = zeros(size(X_All, 3), 3*18);
X_density = zeros(size(X_All, 3), 2);
X_EBC = zeros(size(X_All, 3), 18*18);
X_core = zeros(size(X_All, 3), 19);
X_author = zeros(size(X_All, 3), 18);
X_hubs = zeros(size(X_All, 3), 18);
X_outclose = zeros(size(X_All, 3), 18);
X_inclose = zeros(size(X_All, 3), 18);
X_page = zeros(size(X_All, 3), 18);
X_Efficient = zeros(size(X_All, 3), 19);
X_clust = zeros(size(X_All, 3), 18);
X_Erange = zeros(size(X_All, 3), 2);
X_flow = zeros(size(X_All, 3), 18*2+1);
X_jdegree = zeros(size(X_All, 3), 3);
X_matching_ind = zeros(size(X_All, 3), 18*18*3);
X_modularity = zeros(size(X_All, 3), 19);
X_motif3funct_bin = zeros(size(X_All, 3), 13*18+13);
X_motif3struct_bin = zeros(size(X_All, 3), 13*18+13);
X_motif4funct_bin = zeros(size(X_All, 3), 18*199+199);
X_motif4struct_bin = zeros(size(X_All, 3), 18*199+199);
X_subcen = zeros(size(X_All, 3), 18);
X_tran = zeros(size(X_All, 3), 1);
for i = 1:size(X_AllBin, 3)
    i %#ok<NOPTS> 
    A = X_AllBin(:, :, i);
    r1 = assortativity_bin(A, 1);
    r2 = assortativity_bin(A, 2);
    r3 = assortativity_bin(A, 3);
    r4 = assortativity_bin(A, 4);
    X_assort(i, :) = [r1, r2, r3, r4];
    X_BC(i, :) = betweenness_bin(A);
    [X_degree(i, 1:18), X_degree(i, 19:36), X_degree(i, 37:54)] = degrees_dir(A);
    [X_density(i, 1), ~, X_density(i, 2)] = density_dir(A);
    X_EBC(i, :) = reshape(edge_betweenness_bin(A), 1, 324);
    [X_core(i, 1:18), X_core(i, 19)] = core_periphery_dir(A);
    X_author(i, :) = centrality(digraph(A), "authorities");
    X_hubs(i, :) = centrality(digraph(A), "hubs");
    X_outclose(i, :) = centrality(digraph(A), "outcloseness");
    X_inclose(i, :) = centrality(digraph(A), "incloseness");
    X_page(i, :) = centrality(digraph(A), "pagerank");
    X_Efficient(i, 1:18) = efficiency_wei(A, 1);
    X_Efficient(i, 19) = efficiency_wei(A, 0);
    X_clust(i, :) = clustering_coef_bd(A);
    [~, X_Erange(i, 1), ~, X_Erange(i, 2)] = erange(A); 
    [X_flow(i, 1:18), X_flow(i, 19), X_flow(20:37)] = flow_coef_bd(A);
    [~, X_jdegree(i, 1), X_jdegree(i, 2), X_jdegree(i, 3)] = jdegree(A);
    [Min,Mout,Mall] = matching_ind(A);
    X_matching_ind(i, :) = [Min(:); Mout(:); Mall(:)];
    [X_modularity(i, 1:18), X_modularity(i, 19)] = modularity_dir(A);
    [f, F] = motif3funct_bin(A); X_motif3funct_bin(i, :) = [f(:); F(:)];
    [f, F] = motif3struct_bin(A); X_motif3struct_bin(i, :) = [f(:); F(:)];
    [f, F] = motif4funct_bin(A); X_motif4funct_bin(i, :) = [f(:); F(:)];
    [f, F] = motif4struct_bin(A); X_motif4struct_bin(i, :) = [f(:); F(:)];
    X_subcen(i, :) = subgraph_centrality(double(A));
    X_tran(i, :) = transitivity_bd(A);
end
%%
XG = [X_assort, X_BC, X_degree, X_density, X_EBC, X_core, X_author, ...
      X_hubs, X_outclose, X_inclose, X_page, X_Efficient, X_clust, ...
      X_Erange, X_flow, X_jdegree, X_matching_ind, ...
      X_modularity, X_motif3funct_bin, X_motif3struct_bin, X_motif4funct_bin, ...
      X_motif4struct_bin, X_subcen, X_tran]';
% remove all zero features
idx = ones(size(XG, 1), 1);
for i = 1:size(XG, 1)
    if all(XG(i, :) == 0); idx(i) = 0; end 
end
XG = XG(idx == 1, :);
% normalize features
[XGT,PS] = mapstd(XG(:, 1:nT));
XGTest = mapstd('apply',XG(:, nT+1:end), PS);
XGT = XGT';
XGTest = XGTest';
%%
% regression with 5-Fold cross validation
clear; close all; clc;
load("data\s5_bd_graph_features.mat")
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
SVMModel = fitcsvm(XGTR, yTrain, "KernelFunction", "rbf", "OptimizeHyperparameters","auto", ...
    "HyperparameterOptimizationOptions",struct("AcquisitionFunctionName", 'expected-improvement-plus','ShowPlots',true));

yTrain_predict = predict(SVMModel, XGTR);
figure;
str = "Train Data (SVM)";
sgtitle(str, "Interpreter","latex")
cm = confusionchart(yTrain, yTrain_predict);
cm.Normalization = 'row-normalized';
yTest_predict = predict(SVMModel, XGTestR);
figure;
str = "Test Data (SVM)";
sgtitle(str, "Interpreter","latex")
cm = confusionchart(yTest, yTest_predict);
cm.Normalization = 'row-normalized';























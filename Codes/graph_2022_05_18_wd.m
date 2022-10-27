clear; close all; clc;
load("data\s1_gc.mat")
%% normalize
N = size(XT_gc, 3);
X = cat(3, XT_gc, XTest_gc);
for i = 1:size(X, 3)
    X(:, :, i) = (X(:, :, i) - diag(diag(X(:, :, i)))) / max(X(:, :, i), [], "all");
end
%% Test
% clc
% i = 100;
W_test = X(:, :, 2);
% L_test = weight_conversion(W_test, 'lengths');
% P_test = digraph(W_test);
% D = distance_wei(L_test);
temp = clustering_coef_wd(W_test)
%% extract features from wighted directed matrix
X_clust = zeros(size(X, 3), 18);        % Clustering coefficient
X_core = zeros(size(X, 3), 19);         % Core/periphery structure and core-ness statistic
X_assort = zeros(size(X, 3), 1);        % Assortativity coefficient
X_community = zeros(size(X, 3), 19);    % Optimal community structure
X_Efficient = zeros(size(X, 3), 19);    % Global efficiency, local efficiency
X_modularity = zeros(size(X, 3), 19);   % Optimal community structure and modularity
X_transitivity = zeros(size(X, 3), 1);  % Transitivity
X_BC = zeros(size(X, 3), 18);           % Node betweenness centrality
X_charpath = zeros(size(X, 3), 21);     % network characteristic path length, nodal eccentricity, network radius, network diameter
X_Ediff = zeros(size(X, 3), 18*18+1);   % Global mean and pair-wise diffusion efficiency
X_EBC = zeros(size(X, 3), 18*18);       % Edge betweenness centrality
X_MFPT = zeros(size(X, 3), 18*18);      % Mean first passage time
X_strngth = zeros(size(X, 3), 3*18);    % In-strength and out-strength
X_shpath = zeros(size(X, 3), 18*18);    % distance (shortest weighted path) matrix
for i = 1:size(X, 3)
%     if rem(i, 100) == 0
        i %#ok<NOPTS> 
%     end
    W = X(:, :, i);
    L = weight_conversion(W, 'lengths');
    D = distance_wei(L);
    X_clust(i, :) = clustering_coef_wd(W);
    [X_core(i, 1:18), X_core(i, 19)] = core_periphery_dir(W);
    X_assort(i) = assortativity_wei(W,1);
    [X_community(i, 1:18), X_community(i, 19)] = community_louvain(W);
    X_Efficient(i, 1:18) = efficiency_wei(W, 1);
    X_Efficient(i, 19) = efficiency_wei(W, 0);
    [X_modularity(i, 1:18), X_modularity(i, 19)] = modularity_dir(W);
    X_transitivity(i) = transitivity_wd(W);
    X_BC(i, :) = betweenness_wei(L);
    [lambda,~,ecc,radius,diameter] = charpath(D);
    X_charpath(i, :) = [lambda, ecc', radius, diameter];
    [GEdiff,Ediff] = diffusion_efficiency(W);
    X_Ediff(i, :) = [GEdiff, Ediff(:)'];
    X_EBC(i, :) = reshape(edge_betweenness_wei(L), 1, 324);
    X_MFPT(i, :) = reshape(mean_first_passage_time(W), 1, 324);
    [X_strngth(i, 1:18), X_strngth(i, 19:36), X_strngth(i, 37:54)] = strengths_dir(W);
    X_shpath(i, :) = reshape(D, 1, 324);
end
%% refine features
XG = [X_clust, X_core, X_assort, X_community, X_Efficient, X_modularity,...
      X_transitivity, X_BC, X_charpath, X_Ediff, X_EBC, X_MFPT, X_strngth, X_shpath]';
idx = ones(size(XG, 1), 1);
for i = 1:size(XG, 1)
    if all(XG(i, :) == 0); idx(i) = 0; end 
end
XG = XG(idx == 1, :);
% normalize features
[XGT,PS] = mapstd(XG(:, 1:N));
XGTest = mapstd('apply',XG(:, N+1:end), PS);
XGT = XGT';
XGTest = XGTest';
%% lasso
% regression with 5-Fold cross validation
clear; close all; clc;
load("data\s5_wd_graph_features.mat")
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








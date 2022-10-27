%% load data
clear; close all; clc;
load("data\s5_data.mat")
XT_Corr = adj_mat(XT_corr);
XTest_Corr = adj_mat(XTest_corr);
nT = size(XT_Corr, 3);
X_All = cat(3, XT_Corr, XTest_Corr);
X_Allabs = abs(X_All);
%% test
W = X_Allabs(:, :, 110);
L = weight_conversion(W, 'lengths');
D = distance_wei(L);

%% extract features from binary graph
X_BC = zeros(size(X_Allabs, 3), 18);           % Node betweenness centrality
X_clust = zeros(size(X_Allabs, 3), 18*2+2);    % clustering coefficient
X_community = zeros(size(X_Allabs, 3), 19);    % Optimal community structure
X_assort = zeros(size(X_Allabs, 3), 1);        % Assortativity coefficient
X_Efficient = zeros(size(X_Allabs, 3), 19);
X_transitivity = zeros(size(X_Allabs, 3), 1);  % Transitivity
X_modularity = zeros(size(X_Allabs, 3), 19);   % Optimal community structure and modularity
X_charpath = zeros(size(X_Allabs, 3), 21);     % network characteristic path length, nodal eccentricity, network radius, network diameter
X_Ediff = zeros(size(X_Allabs, 3), 18*18+1);   % Global mean and pair-wise diffusion efficiency
X_EBC = zeros(size(X_Allabs, 3), 18*18);       % Edge betweenness centrality
X_MFPT = zeros(size(X_Allabs, 3), 18*18);      % Mean first passage time
X_strngth = zeros(size(X_Allabs, 3), 3*18);    % In-strength and out-strength
X_shpath = zeros(size(X_Allabs, 3), 18*18);    % distance (shortest weighted path) matrix
X_edge = zeros(size(X_Allabs, 3), 18*18);
X_matching_ind = zeros(size(X_All, 3), 18*18*3);
X_motif3funct_wei = zeros(size(X_All, 3), 13*18*3);
X_motif3struct_wei = zeros(size(X_All, 3), 13*18*3);
X_motif4funct_wei = zeros(size(X_All, 3), 199*18*3);
X_motif4struct_wei = zeros(size(X_All, 3), 199*18*3);
for i = 1:size(X_Allabs, 3)
    i %#ok<NOPTS> 
    W = X_Allabs(:, :, i);
    L = weight_conversion(W, 'lengths');
    D = distance_wei(L);
    X_BC(i, :) = betweenness_wei(L);
    [X_clust(i, 1:18), X_clust(i, 19:36), X_clust(i, 37), X_clust(i, 38)] = clustering_coef_wu_sign(W);
    [X_community(i, 1:18), X_community(i, 19)] = community_louvain(W, [], [], 'negative_asym');
    X_assort(i) = assortativity_wei(W,1);
    X_Efficient(i, 1:18) = efficiency_wei(W, 1);
    X_Efficient(i, 19) = efficiency_wei(W, 0);
    X_transitivity(i) = transitivity_wu(W);
    [X_modularity(i, 1:18), X_modularity(i, 19)] = modularity_und(W);
    [lambda,~,ecc,radius,diameter] = charpath(D);
    X_charpath(i, :) = [lambda, ecc', radius, diameter];
    [GEdiff,Ediff] = diffusion_efficiency(W);
    X_Ediff(i, :) = [GEdiff, Ediff(:)'];
    X_EBC(i, :) = reshape(edge_betweenness_wei(L), 1, 324);
    X_MFPT(i, :) = reshape(mean_first_passage_time(W), 1, 324);
    [X_strngth(i, 1:18), X_strngth(i, 19:36), X_strngth(i, 37:54)] = strengths_dir(W);
    X_shpath(i, :) = reshape(D, 1, 324);
    X_edge(i, :) = reshape(X_All(:, :, i), 1, 324);
    [Min,Mout,Mall] = matching_ind(W);
    X_matching_ind(i, :) = [Min(:); Mout(:); Mall(:)];
    [I,Q,F]=motif3funct_wei(W); X_motif3funct_wei(i, :) = [I(:); Q(:); F(:)];
    [I,Q,F]=motif3struct_wei(W); X_motif3struct_wei(i, :) = [I(:); Q(:); F(:)];
    [I,Q,F]=motif4funct_wei(W); X_motif4funct_wei(i, :) = [I(:); Q(:); F(:)];
    [I,Q,F]=motif4struct_wei(W); X_motif4struct_wei(i, :) = [I(:); Q(:); F(:)];
end
%% refine features
XG = [X_clust, X_assort, X_community, X_Efficient, X_modularity, X_transitivity, ...
      X_BC, X_charpath, X_Ediff, X_EBC, X_MFPT, X_strngth, X_shpath, X_edge, ...
      X_matching_ind, X_motif3funct_wei, X_motif3struct_wei, X_motif4funct_wei, X_motif4struct_wei]';
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
% save("s5_wu_graph_features", "XGT", "XGTest")
%% lasso
% regression with 5-Fold cross validation
clear; close all; clc;
load("data\s5_wu_graph_features.mat")
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










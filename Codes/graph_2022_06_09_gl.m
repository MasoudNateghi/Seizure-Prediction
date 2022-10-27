clear; close all; clc;
load("data\s1_data.mat")
%% correlation 
% for p = linspace(0.1, 0.9, 9)
p = 0.4;
%%
X_AllCorr = [XT_corr; XTest_corr];
X_AllPlv = [XT_plv; XTest_plv];
nT = size(XT_corr, 1);
% thresholding
corrMax = max(abs(X_AllCorr(:)));
Gthr = p*corrMax;
X_AllCorrBin = abs(X_AllCorr) >= Gthr;
A = binGraph(X_AllCorrBin(5, :));
plotGraph(A)
%% 
XG = zeros(size(X_AllCorr, 1), 15);
for i = 1:size(X_AllCorr, 1)
    i %#ok<NOPTS>
    A = binGraph(X_AllCorrBin(i, :));
    [XG(i, 1:4), XG(i, 5:end)] = graphletCounting(A);
end
XG = XG'; 
[XGT,PS] = mapstd(XG(:, 1:nT));
XGTest = mapstd('apply',XG(:, nT+1:end), PS);
XGT = XGT';
XGTest = XGTest';
%% lasso
% regression with 5-Fold cross validation
[B,FitInfo] = lasso(XGT,yTrain,'CV',5);
idxLambda1SE = FitInfo.Index1SE;
coef = B(:,idxLambda1SE);
coef0 = FitInfo.Intercept(idxLambda1SE);
% plot MSE for different lambda values
lassoPlot(B,FitInfo,'PlotType','CV');
legend('show') % Show legend
% AUC and confusion matrix for train data
yhatTrain = XGT*coef + coef0;
[X,Y,T,AUC,OPTROCPT] = perfcurve(yTrain,yhatTrain,1); 
% plot(X, Y); xlabel('fpr'); ylabel('tpr'); title('AUC for train data (corr)')
fprintf('AUC for train data = %f\n', AUC)
best_thr_index = find(X == OPTROCPT(1) & Y == OPTROCPT(2));
best_thr = T(best_thr_index); 
yTrain_predict = zeros(size(yTrain));
yTrain_predict(yhatTrain >= best_thr) = 1;
figure;
str = "Train Data with thr = " + num2str(p);
sgtitle(str, "Interpreter","latex")
cm = confusionchart(yTrain, yTrain_predict);
cm.Normalization = 'row-normalized';
% figure;
% cm = confusionchart(yTrain, yTrain_predict);  %#ok<NASGU> 
% sgtitle(str, "Interpreter","latex")
% AUC and confusion matrix for test data
yhatTest = XGTest*coef + coef0;
yTest_predict = zeros(size(yTest));
yTest_predict(yhatTest >= best_thr) = 1;
figure;
cm = confusionchart(yTest, yTest_predict);
str = "Test Data with thr = " + num2str(p);
sgtitle(str, "Interpreter","latex")
cm.Normalization = 'row-normalized';
% figure;
% cm = confusionchart(yTest, yTest_predict); 
% sgtitle(str, "Interpreter", "latex")
% end 
clear; close all; clc;
load("data\s5_trials.mat")
%%
% m = 2;
% n = 1;
% [F,~] = granger_cause(XTrain(18, :, 1),XTrain(18, :, 1),0.01,5)
XT_gc = zeros(18, 18, size(XTrain, 3));
for i = 1:size(XTrain, 3)
    i
    for m = 1:18
        for n = 1:18
            XT_gc(m, n, i) = granger_cause(XTrain(m, :, i),XTrain(n, :, i),0.01,5);
        end
    end
end
%%
XTest_gc = zeros(18, 18, size(XTest, 3));
for i = 1:size(XTest, 3)
    i
    for m = 1:18
        for n = 1:18
            XTest_gc(m, n, i) = granger_cause(XTest(m, :, i),XTest(n, :, i),0.01,5);
        end
    end
end

%%
%% reshape
clear; close all; clc;
load("data\s1_gc.mat")

XT = zeros(size(XT_gc, 3), 18*18);
for i = 1:size(XT_gc, 3)
    XT(i, :) = reshape(XT_gc(:, :, i), 1, 18*18);
end
XTest_GC = zeros(size(XTest_gc, 3), 18*18);
for i = 1:size(XTest_gc, 3)
    XTest_GC(i, :) = reshape(XTest_gc(:, :, i), 1, 18*18);
end
%% train lasso
[B,FitInfo] = lasso(XT,yTrain,'CV',5);
idxLambda1SE = FitInfo.Index1SE;
coef = B(:,idxLambda1SE);
coef0 = FitInfo.Intercept(idxLambda1SE);
lassoPlot(B,FitInfo,'PlotType','CV');
legend('show') % Show legend
% plot MSE for different lambda values
lassoPlot(B,FitInfo,'PlotType','CV');
legend('show') % Show legend
% AUC and confusion matrix for train data
yhatTrain = XT*coef + coef0;
[X,Y,T,AUC,OPTROCPT] = perfcurve(yTrain,yhatTrain,1); 
% plot(X, Y); xlabel('fpr'); ylabel('tpr'); %title('AUC for train data (corr)')
% fprintf('AUC for train data = %f\n', AUC)
% best_thr_index = find(X == OPTROCPT(1) & Y == OPTROCPT(2));
% best_thr = T(best_thr_index); 
best_thr = find_best_thr(yTrain, yhatTrain, T);
yTrain_predict = zeros(size(yTrain));
yTrain_predict(yhatTrain >= best_thr) = 1;
figure;
str = "Train Data";
sgtitle(str, "Interpreter","latex")
cm = confusionchart(yTrain, yTrain_predict);
cm.Normalization = 'row-normalized';
yhatTest = XTest_GC*coef + coef0;
yTest_predict = zeros(size(yTest));
yTest_predict(yhatTest >= best_thr) = 1;
figure;
cm = confusionchart(yTest, yTest_predict);
str = "Test Data";
sgtitle(str, "Interpreter","latex")
cm = confusionchart(yTest, yTest_predict);
cm.Normalization = 'row-normalized';
%% svm
best_features = find(coef ~= 0);
XGTR = XT(:, best_features); % reduced
XGTestR = XTest_GC(:, best_features);
SVMModel = fitcsvm(XGTR, yTrain, "KernelFunction", "rbf", "OptimizeHyperparameters","auto", ...
    "HyperparameterOptimizationOptions",struct("AcquisitionFunctionName", 'expected-improvement-plus','ShowPlots',true));

yTrain_predict = predict(SVMModel, XGTR);
figure;
str = "Train Data";
sgtitle(str, "Interpreter","latex")
cm = confusionchart(yTrain, yTrain_predict);
cm.Normalization = 'row-normalized';
yTest_predict = predict(SVMModel, XGTestR);
figure;
str = "Test Data";
sgtitle(str, "Interpreter","latex")
cm = confusionchart(yTest, yTest_predict);
cm.Normalization = 'row-normalized';























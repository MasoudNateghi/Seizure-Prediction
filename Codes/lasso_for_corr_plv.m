%% load data
clear; close all; clc;
load("inter_ictal_s5.mat");
load("pre_ictal_s5.mat");
fs = 256;
%% shuffle inter-ictal data
rng('default')
all_trials = cat(3, inter_ictal, pre_ictal); % concat all trials
n_inter_ictal = size(inter_ictal, 3); % number of inter-ictal trials
n_pre_ictal = size(pre_ictal, 3); % number of pre-ictal trials
n_all_trials = n_inter_ictal + n_pre_ictal; % number of all trials
% create labels
% pre-ictal = 1, inter-ictal = 0
labels = [zeros(n_inter_ictal, 1); ones(n_pre_ictal, 1)];
% shuffle all trials
p = randperm(n_all_trials); % create permutation vector
all_trials_shuff = all_trials(:, :, p); % shuffle data along third dimension
labels_shuff = labels(p); % shuffle labels
%% separate train and test data
c = cvpartition(n_all_trials,'HoldOut',0.3); % keep 30% of data for test
idxTrain = training(c); % training data indices
idxTest = ~idxTrain; % test data indices
trialsTrain = all_trials_shuff(:, :, idxTrain); % train data
XTest  = all_trials_shuff(:, :, idxTest ); % test data
labelTrain = labels_shuff(idxTrain); % train labels
yTest  = labels_shuff(idxTest ); % test labels
%% handle imbalanced dataset (downsample)
p_index = labelTrain == 1; % separate train pre-ictal indices
n_index = labelTrain == 0; % separate train inter-ictal indices
N_p = sum(p_index); % number of train pre-ictal trials
N_n = sum(n_index); % number of train inter-ictal trials
trialTrain_p = trialsTrain(:, :, p_index); % separate train pre-ictal trials
trialTrain_n = trialsTrain(:, :, n_index); % separate train inter-ictal trials
p = randperm(N_n); % create shuffle vector 
all_trials_d = cat(3, trialTrain_p, trialTrain_n(:, :, p(1:N_p))); % pick same number of inter-ictal trials as pre-ictal
labels_d = [ones(N_p, 1); zeros(N_p, 1)]; % create the labels again
p = randperm(2*N_p); % create shuffle vector
XTrain = all_trials_d(:, :, p); % shuffle trials
yTrain = labels_d(p); % shuffle labels
%% calculate correlation 
L_window = 2; % length of window(2s)
overlap = 0.9; % overlap of windows(90%)
XT_corr = jcorr(XTrain, fs, L_window, overlap);
XTest_corr = jcorr(XTest, fs, L_window, overlap);
%% load data
clear; close all; clc;
load("data\s1_data.mat")
% regression with 5-Fold cross validation
[B,FitInfo] = lasso(XT_corr,yTrain,'CV',5);
idxLambda1SE = FitInfo.Index1SE;
coef = B(:,idxLambda1SE);
coef0 = FitInfo.Intercept(idxLambda1SE);
% plot MSE for different lambda values
lassoPlot(B,FitInfo,'PlotType','CV');
legend('show') % Show legend
% AUC and confusion matrix for train data
yhatTrain = XT_corr*coef + coef0;
[X,Y,T,AUC,OPTROCPT] = perfcurve(yTrain,yhatTrain,1); 
plot(X, Y); xlabel('fpr'); ylabel('tpr'); title('AUC for train data (corr)')
fprintf('AUC for train data = %f\n', AUC)
best_thr_index = find(X == OPTROCPT(1) & Y == OPTROCPT(2));
best_thr = T(best_thr_index); %#ok<FNDSB> 
yTrain_predict = zeros(size(yTrain));
yTrain_predict(yhatTrain >= best_thr) = 1;
figure;
str = "Train Data";
sgtitle(str)
cm = confusionchart(yTrain, yTrain_predict);
cm.Normalization = 'row-normalized';
% figure;
% cm = confusionchart(yTrain, yTrain_predict); %#ok<NASGU> 
% AUC and confusion matrix for test data
yhatTest = XTest_corr*coef + coef0;
yTest_predict = zeros(size(yTest));
yTest_predict(yhatTest >= best_thr) = 1;
figure;
str = "Test Data";
sgtitle(str)
cm = confusionchart(yTest, yTest_predict);
cm.Normalization = 'row-normalized';
% figure;
% cm = confusionchart(yTest, yTest_predict); %#ok<NASGU> 
%% calculate plv
L_window = 2;
overlap = 0.9;
XT_plv = jplv(XTrain, fs, L_window, overlap);
XTest_plv = jplv(XTest, fs, L_window, overlap);
%% save data
% save("s5_data.mat", "XTest_plv", "XT_plv", "XT_corr", "XTest_corr", "yTest", "yTrain")
%% regression with 5-Fold cross validation
[B,FitInfo] = lasso(XT_plv,yTrain,'CV',5);
idxLambda1SE = FitInfo.Index1SE;
coef = B(:,idxLambda1SE);
coef0 = FitInfo.Intercept(idxLambda1SE);
% plot MSE for different lambda values
lassoPlot(B,FitInfo,'PlotType','CV');
legend('show') % Show legend
% AUC and confusion matrix for train data
yhatTrain = XT_plv*coef + coef0;
[X,Y,T,AUC,OPTROCPT] = perfcurve(yTrain,yhatTrain,1); 
plot(X, Y); xlabel('fpr'); ylabel('tpr'); title('AUC for train data (corr)')
fprintf('AUC for train data = %f\n', AUC)
best_thr_index = find(X == OPTROCPT(1) & Y == OPTROCPT(2));
best_thr = T(best_thr_index); 
yTrain_predict = zeros(size(yTrain));
yTrain_predict(yhatTrain >= best_thr) = 1;
figure;
str = "Train Data";
sgtitle(str)
cm = confusionchart(yTrain, yTrain_predict);
cm.Normalization = 'row-normalized';
% figure;
% cm = confusionchart(yTrain, yTrain_predict); %#ok<NASGU>  
% AUC and confusion matrix for train data
yhatTest = XTest_plv*coef + coef0;
yTest_predict = zeros(size(yTest));
yTest_predict(yhatTest >= best_thr) = 1;
figure;
str = "Test Data";
sgtitle(str)
cm = confusionchart(yTest, yTest_predict);
cm.Normalization = 'row-normalized';
% figure;
% cm = confusionchart(yTest, yTest_predict); 







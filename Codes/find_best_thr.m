function [best_thr, best_acc] = find_best_thr(yTrain, yhatTrain, T)
best_acc = 0;
for i = 1:length(T)
    y_predict = yhatTrain >= T(i);
    acc = sum(y_predict == yTrain) / length(yTrain);
    if acc >= best_acc
        best_acc = acc;
        best_thr = T(i);
    end
end
function thr = findMaxThreshold(tree, X)
thr = inf;
for m = 1:18
    for n = m+1:18
        if tree(m, n) == 1 && abs(X(m, n)) < thr
            thr = abs(X(m, n));
        end
    end
end
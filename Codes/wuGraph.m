function A = wuGraph(X)
counter = 1;
A = zeros(18, 18);
for m = 1:18
    for n = m + 1:18
        A(m, n) = X(counter);
        A(n, m) = X(counter);
        counter = counter + 1;
    end
end

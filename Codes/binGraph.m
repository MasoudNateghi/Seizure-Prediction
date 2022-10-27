function A = binGraph(X)
counter = 1;
A = zeros(18, 18);
for m = 1:18
    for n = 1:18
        if n <= m; continue; end
        if X(counter) == 1
            A(m, n) = 1;
            A(n, m) = 1;
        end
        counter = counter + 1;
    end
end

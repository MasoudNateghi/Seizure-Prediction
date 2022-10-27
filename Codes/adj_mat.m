function XT_corr = adj_mat(trials)
N = size(trials, 1);
XT_corr = zeros(18, 18, N);
for i = 1:N
    temp = zeros(18, 18);
    count = 1;
    for m = 1:18
        for n = 1:18
            if n <= m, continue, end
            temp(m, n) = trials(i, count);
            temp(n, m) = trials(i, count);
            count = count + 1;
        end
    end
    XT_corr(:, :, i) = temp;
end
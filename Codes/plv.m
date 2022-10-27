function vals = plv(X)
m = size(X, 1);     % number of channels
n = size(X, 2);     % number of samples
phase = zeros(m, n);
for i = 1:m
    phase(i, :) = angle(hilbert(X(i, :)));
end
vals = zeros(m);
for i = 1:m
    for j = 1:m
        phi_x = phase(i, :);
        phi_y = phase(j, :);
        phi_xy = phi_x - phi_y;
        vals(i, j) = abs(1 / n * sum(exp(1i*phi_xy)));
    end
end

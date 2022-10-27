%%
clear; clc;
fs = 256;
pre_ictal = zeros(18, 1280, 30240);
n_previous = 0;
L = 1280;
%%
second1 = 1862 - 300;
second2 = 1862;
index1 = second1 * fs + 1
index2 = second2 * fs
n = floor((index2 - index1 + 1)/L)
temp_sel = temp(:, index1:index1-1+n*L);
%%
for i=1:n
    pre_ictal(:, :, i+n_previous) = temp_sel(:, 1+(i-1)*L:i*L);
end
n_previous = n_previous + n
%%
clear temp temp_sel



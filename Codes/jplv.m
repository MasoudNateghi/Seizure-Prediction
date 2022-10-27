function plv_feature = jplv(trials, fs, L_window, overlap)
n_trials = size(trials, 3);
n_channel = size(trials, 1);
n_feature = n_channel*(n_channel - 1)/2;
trial_length = size(trials, 2) / fs;
n_window = (trial_length - L_window) / ((1-overlap)*L_window) + 1;
noverlap = floor((1-overlap)*L_window*fs); 
L = 2*fs;
plv_feature = zeros(n_trials, n_feature);
% 2 of 18 = 153
% 16 windows
% 16*153 = 2448 features
for k = 1:n_trials
    k
    average = zeros(1, n_feature);
    for i = 1:n_window
        start_index = noverlap*(i-1)+1;
        end_index   = noverlap*(i-1)+L;
        windowed = trials(:, start_index:end_index, k);
        vals = plv(windowed);
        count = 1;
        for m = 1:18
            for n = 1:18
                if n <= m
                    continue
                end
                average(count) = average(count) + vals(m, n)/16;
                count = count + 1;
            end
        end
    end
    plv_feature(k, :) = average;
end
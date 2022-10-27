%% load data
clear; close all; clc;
fs = 256;
files = dir('*.mat');
for i = 1:length(files)
    load(files(i).name)
    nChannel = size(val, 1);
    filtered_signal = zeros(size(val));
    for k = 1:nChannel
        filtered_signal(k, :) = bandpass(val(k, :), [1 45], fs);
    end
    name = files(i).name;
    name_split = split(name, '.');
    name_split{1} = strcat(name_split{1}, '_filtered');
    name = strcat(name_split{1}, '.', name_split{2});
    save(name, "filtered_signal")
end






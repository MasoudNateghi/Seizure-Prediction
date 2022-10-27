clear; close all; clc;
files = dir('*.mat');
for i = 1:length(files)
    load(files(i).name)
    temp = zeros(18, length(filtered_signal));
    temp(1, :) = filtered_signal(1, :);                         % FP1
    temp(2, :) = -filtered_signal(2, :);                        % T7
    temp(3, :) = temp(2, :) - filtered_signal(3, :);            % P7
    temp(4, :) = temp(3, :) - filtered_signal(4, :);            % O1
    temp(5, :) = temp(1, :) - filtered_signal(5, :);            % F3
    temp(6, :) = temp(5, :) - filtered_signal(6, :);            % C3
    temp(7, :) = temp(6, :) - filtered_signal(7, :);            % P3
    temp(8, :) = temp(2, :) - filtered_signal(20, :);           % FT9
    temp(9, :) = temp(8, :) - filtered_signal(21, :);           % FT10
    temp(10, :) = temp(9, :) - filtered_signal(22, :);          % T8
    temp(11, :) = temp(10, :) - filtered_signal(23, :);         % P8
    temp(12, :) = temp(11, :) - filtered_signal(16, :);         % O2
    temp(13, :) = temp(10, :) + filtered_signal(14, :);         % F8
    temp(14, :) = temp(13, :) + filtered_signal(13, :);         % FP2
    temp(15, :) = temp(12, :) + filtered_signal(12, :);         % P4
    temp(16, :) = temp(15, :) + filtered_signal(11, :);         % C4
    temp(17, :) = temp(16, :) + filtered_signal(10, :);         % F4
    temp(18, :) = temp(18, :);                                  % F7
    temp = temp - mean(temp, 1);
    name = files(i).name;
    name_split = split(name, '_');
    name = strcat(name_split{1}, '_', name_split{2}, '_', name_split{3} ...
        , '_CAR.mat');
    save(name, "temp");
end










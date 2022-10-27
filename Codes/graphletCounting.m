function [g3, g4] = graphletCounting(A)
N_TT = 0; N_SuSv = 0; N_TSuAndSv = 0; N_SS = 0;
N_TI = 0; N_SuAndSvI = 0; N_II = 0; N_II1 = 0;
% N_SvSv = 0; N_SuSu = 0;
X = zeros(1, length(A));
g3 = zeros(1, 4);
g4 = zeros(1, 11); 
for u = 1:length(A)
    for v = u+1:length(A)
        if A(u, v) == 0; continue; end
        Nu = find(A(u, :) == 1); 
        Nv = find(A(v, :) == 1);
        Star_u = zeros(1, length(A)); 
        Star_v = zeros(1, length(A)); 
        Tri_e = zeros(1, length(A));
        for w = Nu
            if w == v; continue; end
            Star_u(w) = 1;
            X(w) = 1;
        end
        for w = Nv
            if w == u; continue; end
            if X(w) == 1
                Tri_e(w) = 1;
                X(w) = 2;
                Star_u(w) = 0;
            else
                Star_v(w) = 1;
                X(w) = 3;
            end
        end
        g3(1) = g3(1) + sum(Tri_e ~= 0);
        g3(2) = g3(2) + sum(Star_u ~= 0) + sum(Star_v ~= 0);
        g3(3) = g3(3) + length(A) - length(unique([Nu Nv]));
        % Get Counts of 4-Cliques & 4-Cycles
        % procedure CLIQUECOUNT
        for w = find(Tri_e ~= 0)
            Nw = find(A(w, :) == 1);
            for r = Nw
                if X(r) == 2; g4(1) = g4(1) + 1; end
            end
            X(w) = 0;
        end
        % procedure CYCLECOUNT
        for w = find(Star_u ~= 0)
            Nw = find(A(w, :) == 1);
            for r = Nw
                if X(r) == 3; g4(4) = g4(4) + 1; end
            end
            X(w) = 0;
        end
        % Get Unrestricted Counts for 4-Node Connected Graphlets
        if sum(Tri_e ~= 0) >= 2
            N_TT = N_TT + nchoosek(sum(Tri_e ~= 0), 2);
        end
        N_SuSv = N_SuSv + sum(Star_u ~= 0) * sum(Star_v ~= 0);
        N_TSuAndSv = N_TSuAndSv + sum(Tri_e ~= 0) * (sum(Star_u ~= 0) + sum(Star_v ~= 0));
        if sum(Star_u ~= 0) >= 2
            N_SuSu = nchoosek(sum(Star_u ~= 0), 2);
        else
            N_SuSu = 0;
        end
        if sum(Star_v ~= 0) >= 2
            N_SvSv = nchoosek(sum(Star_v ~= 0), 2);
        else
            N_SvSv = 0;
        end
        N_SS = N_SS + N_SuSu + N_SvSv;
        % Get Unrestricted Counts for 4-Node Disconnected Graphlets
        N_TI = N_TI + sum(Tri_e ~= 0) * (length(A) - length(unique([Nu Nv])));
        N_SuI = sum(Star_u ~= 0) * (length(A) - length(unique([Nu Nv])));
        N_SvI = sum(Star_v ~= 0) * (length(A) - length(unique([Nu Nv])));
        N_SuAndSvI = N_SuAndSvI + N_SuI + N_SvI;
        if length(A) - length(unique([Nu Nv])) >= 2
            N_II = N_II + nchoosek((length(A) - length(unique([Nu Nv]))), 2);
        end
        N_II1 = N_II1 + sum(A, "all")/2 - (length(Nu) - 1) - (length(Nv) - 1) - 1;
        for w = Nv; X(w) = 0; end
    end 
end
g3(1) = g3(1) / 3;
g3(2) = g3(2) / 2;
g3(4) = nchoosek(length(A), 3) - g3(1) - g3(2) - g3(3);
g4(1) = g4(1) / 6;
g4(4) = g4(4) / 4;
g4(2) = N_TT - 6 * g4(1);
g4(6) = N_SuSv - 4 * g4(4);
g4(3) = N_TSuAndSv / 2 - 2 * g4(2);
g4(5) = N_SS / 3 - g4(3)/3; 
g4(7) = N_TI / 3 - g4(3) / 3;
g4(8) = N_SuAndSvI / 2 - g4(6);
g4(9) = N_II1 / 2;
g4(10) = N_II - g4(9); % modified
g4(11) = nchoosek(length(A), 4) - sum(g4(1:10));

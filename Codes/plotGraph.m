function plotGraph(A, opt)
if opt == 1 
    G = graph(A); 
else 
    G = digraph(A); 
end
load("data\AllElectrodes.mat") %#ok<LOAD> 
elabels = cell(18, 1);
elocsX = zeros(1, 18);
elocsY = zeros(1, 18);
used_elecs = [1, 15, 23, 27, 5, 13, 21, 8, 43, 52, 60, 64, 42, 34, 58, 50, 40, 7];
for i = 1:18
    elabels{i} = AllElectrodes(used_elecs(i)).labels;
    elocsX(i) = AllElectrodes(used_elecs(i)).X;
    elocsY(i) = AllElectrodes(used_elecs(i)).Y;
end
% 90 degree rotation
eloc = [elocsX; elocsY];
Q = [0 1; -1 0];
eloc = Q' * eloc;
G.Nodes.Name = elabels;
figure;
plot(G, 'XData', eloc(1, :), 'YData', eloc(2, :))
axis("equal")
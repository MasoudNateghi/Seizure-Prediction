# Seizure-Prediction
In this repository, directional and bidirectional connectivities were used to predict whether a given time interval is pre-ictal or inter-ictal. 

B.Sc thesis of Masoud Nateghi @Sharif University of technology

Dataset available at: https://physionet.org/content/chbmit/1.0.0/

## Connectivity
Information in connections: Activities in a brain are correlated to a specific mental task, which uses a bunch of neurons for a biological computation that produces an electromagnetic signal. In a system like the human brain, the similarity between these recorded signals (electromagnetic, chemical, etc.) can give us a piece of good information about the functions of the brain and later investigations about related diseases. The similarity criterion and variable normalizations finally form a weight matrix that can be transformed into a weighted graph and, if the criterion is causal, a directional graph. Pearson correlation and PLV were used for weighted\unweighted undirected graphs, whereas granger causality was used for weighted\unweighted directed graphs. Unweighted Graphs were obtained using thresholding on edges' weights.

Then Several properties were extracted from these graphs (weightd\unweighted and directed\undirected) using Brain Connectivity Toolbox(like clustering coefficient, centralities, characteristic path, etc.).

BCT toolbox available at: https://sites.google.com/site/bctnet/

Lasso regression + thresholding was used to classify different intervals. Best results were achieved using granger causality through weighted-directed graphs.

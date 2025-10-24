The VAE model was fist trained. (1) use BERT-based embeddings to convert target texts into vectors (the bert-model can be substitituted based on research needs.) (2) only use the latent vectors (50 dimensions) for classification
Use a decision tree model to classify the latent vectors;
Use the tree model's GINI Importance to identify the important dimensions that contribute to 95% confidence.
Conduct pertubation tests on the identified important dimensions to further confirm the determinate (most important) dimensions. (1) change values of the dimensions from -2 to 2; (2) while the confidence of the tree model to make decisions remains the same, the dimensions that after the pertubation tests causes a change in classification decisions are selected for further analysis.

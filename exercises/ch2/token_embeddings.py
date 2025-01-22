import torch
from torch import nn

input_ids = torch.tensor([2, 3, 5, 1])

vocab_size = 6
output_dim = 3

torch.manual_seed(666)
embedding_layer = nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
print()
print(embedding_layer(torch.tensor([3])))


# ===============================
# Visualize the embeddings
# ===============================
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Get the embedding weights
weights = embedding_layer.weight.detach().numpy()

# Use PCA to reduce dimensions to 2D
pca = PCA(n_components=2)
weights_2d = pca.fit_transform(weights)

# Plot the 2D projection
plt.figure(figsize=(8, 6))
plt.scatter(weights_2d[:, 0], weights_2d[:, 1], c='blue', alpha=0.7)

# Annotate the points with token IDs
for i, coord in enumerate(weights_2d):
    plt.text(coord[0] + 0.02, coord[1] + 0.02, str(i), fontsize=12)

plt.title("2D Projection of Embedding Vectors (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid()
plt.show()
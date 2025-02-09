import torch

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your      x^(1)
        [0.55, 0.87, 0.66],  # journey   x^(2)
        [0.57, 0.85, 0.64],  # starts    x^(3)
        [0.22, 0.58, 0.33],  # with      x^(4)
        [0.77, 0.25, 0.10],  # one       x^(5)
        [0.05, 0.80, 0.55],  # step      x^(6)
    ]
)
print()
print(inputs)

# calculate attention weights for x^(2)
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print()
print(attn_scores_2)

# normalize the attention scores
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())


def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# for better numerical stability
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# calculate the context vector for x^(2)
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
    print(i, x_i, "x", attn_weights_2[i], "=", attn_weights_2[i] * x_i)
print("Context vector for x^(2):", context_vec_2)

# a more simplified version
context_vec_2 = torch.matmul(attn_weights_2, inputs)
print("Context vector for x^(2):", context_vec_2)

# calculate attention scores for all pairs of inputs
attn_scores = torch.empty(inputs.shape[0], inputs.shape[0])
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)

# more efficient version
attn_scores = inputs @ inputs.T
print(attn_scores)

#  normalize with softmax
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

print("All row sums:", attn_weights.sum(dim=-1))

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
print("Previous 2nd context vector:", context_vec_2)
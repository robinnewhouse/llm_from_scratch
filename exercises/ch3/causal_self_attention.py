from SelfAttention import SelfAttention_v2 as SelfAttention
import torch
import matplotlib.pyplot as plt
import time
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

"""
Visualize the attention weights
"""
def visualize_attention(attn_weights):
    plt.clf()
    plot_weights = attn_weights.detach().numpy()

    plt.imshow(plot_weights)
    for i in range(attn_weights.shape[0]):
        for j in range(attn_weights.shape[1]):
            plt.text(j, i, f"{attn_weights[i, j]:.2f}", ha="center", va="center", color="black")
    plt.savefig("exercises/ch3/attn_weights.png")

torch.manual_seed(123)

sa_v2 = SelfAttention(d_in=inputs.shape[1], d_out=2)
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
values = sa_v2.W_value(inputs)

attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
masked_simple = attn_weights*mask_simple
print(masked_simple)
visualize_attention(masked_simple)

row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
visualize_attention(masked_simple_norm)

print(masked_simple_norm)

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
visualize_attention(mask)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
visualize_attention(masked)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
visualize_attention(attn_weights)
print(attn_weights)

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print(dropout(example))
visualize_attention(dropout(example))

visualize_attention(dropout(attn_weights))

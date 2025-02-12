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

x_2 = inputs[1]  # the second input element
d_in = inputs.shape[1]  # the input embedding size, d_in=3
d_out = 2  # the output embedding size, d_out=2

torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)

print("W_query")
print(W_query)
# print(W_key)
# print(W_value)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print("query_2")
print(query_2)
# print(key_2)
# print(value_2)

keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

print(keys)
print(values)

# compute the attention score omega_22
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print("attn_score_22")
print(attn_score_22)

attn_scores_2 = query_2 @ keys.T
print("attn_scores_2")
print(attn_scores_2)

# compute the attention weights
# scale the attention scores by sqrt(d_k)
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print("attn_weights_2")
print(attn_weights_2)
print(attn_weights_2.sum())

context_vec_2 = attn_weights_2 @ values
print("context_vec_2")
print(context_vec_2)

"""
The shape of this
tensor is (b, num_heads,
num_tokens, head_dim)
= (1, 2, 3, 4).
"""
a = torch.tensor(
    [
        [
            [
                [0.2745, 0.6584, 0.2775, 0.8573],
                [0.8993, 0.0390, 0.9268, 0.7388],
                [0.7179, 0.7058, 0.9156, 0.4340],
            ],
            [
                [0.0772, 0.3565, 0.1479, 0.5331],
                [0.4066, 0.2318, 0.4545, 0.9737],
                [0.4606, 0.5159, 0.4220, 0.5786],
            ],
        ]
    ]
)
print(a)
print(a.transpose(2, 3))
print(a @ a.transpose(2, 3))

first_head = a[0, 0, :, :]
first_res = first_head @ first_head.T
print("First head:\n", first_head)
print("First result:\n", first_res)
second_head = a[0, 1, :, :]
second_res = second_head @ second_head.T
print("\nSecond head:\n", second_head)
print("\nSecond result:\n", second_res)




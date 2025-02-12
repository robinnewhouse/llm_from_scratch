import torch.nn as nn
import torch


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)  # keep batch dimension
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec


if __name__ == "__main__":

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

    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape)

    ca = CausalAttention(d_in, d_out, context_length=batch.shape[1], dropout=0.0)
    print("ca.W_query", ca.W_query.weight)
    print("ca.W_key", ca.W_key.weight)
    print("ca.W_value", ca.W_value.weight)
    print("ca(inputs)", ca(batch))

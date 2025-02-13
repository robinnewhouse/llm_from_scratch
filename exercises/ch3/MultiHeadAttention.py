import torch
import torch.nn as nn
from CausalAttention import CausalAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Uses a Linear layer to combine head outputs
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Add a num_heads dimension, unrolling the last dimension
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transposes from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute the dot product for each head
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens].bool()
        # Mask the scores to enforce causality
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # This is where we flatten the last two dimensions num_heads * head_dim => d_out
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

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

    torch.manual_seed(123)

    batch = torch.stack((inputs, inputs), dim=0)
    context_length = batch.shape[1]  # This is the number of tokens
    batch_size, context_length, d_in = batch.shape
    d_out = 2

    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    # print(context_vecs)

    # GPT-2 (small) specifications
    gpt2_d_model = 768  # embedding dimension
    gpt2_num_heads = 12
    gpt2_context_length = 1024
    dropout = 0.1  # typical dropout value for transformer models
    
    # Initialize GPT-2 sized attention module
    gpt2_mha = MultiHeadAttention(
        d_in=gpt2_d_model,
        d_out=gpt2_d_model,
        context_length=gpt2_context_length,
        dropout=dropout,
        num_heads=gpt2_num_heads
    )
    
    # Create a sample input tensor with GPT-2 dimensions
    sample_batch_size = 10
    sample_seq_length = 100  # Using a smaller sequence length for testing
    sample_input = torch.randn(sample_batch_size, sample_seq_length, gpt2_d_model)
    
    # Test the attention module
    output = gpt2_mha(sample_input)
    print(f"\nGPT-2 attention output shape: {output.shape}")



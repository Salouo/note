import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):     # d_model needs to be divisible by n_head for final contenation
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        # QKV
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

    def split_heads(self, x, batch_size):
        # （batch_size, length, d_model(embedding_dim)） -> (batch_size, length, num_heads, d_k)
        x = x.reshape(batch_size, -1, self.nhead, self.d_k)
        x = x.permute(0, 2, 1, 3)   # (batch_size, num_heads, length, d_k)
        return x

    def forward(self, x_q, x_k, x_v, mask=None):
        # When x_q = x_k = x_v, this module is a normal attention module.
        # When x_q = decoder_output, x_k = x_v = encoder_output, and mask is not None, this module is a masked attention module.
        # When x_q = decoder_output, x_k = x_v = encoder_output, and mask is None, this module is a interaction attention module.
        batch_size = x_q.size(0)

        # Get QKV matrices
        q = self.q_linear(x_q)
        k = self.k_linear(x_k)
        v = self.v_linear(x_v)

        # Split QKV matrices based on nhead
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Compute attention scores
        scale_attention = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32)) # (Q @ K.T) / sqrt(d_k)

        # If mask is required, mask the input matrix before doing softmax.
        if mask is not None:
            # Set the elements of the input matrix to -infinity where the corresponding elements in the mask matrix are 0
            scale_attention = scale_attention.masked_fill(mask==0, -torch.inf)

        attention_weights = self.softmax(scale_attention)
        attention_output = torch.matmul(attention_weights, v)   # shape: (batch_size, num_heads, length, attention_features)

        # Concantenate heads
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()     # shape: (batch_size, length, num_heads, attention_features)
        attention_output = attention_output.reshape(batch_size, -1, self.d_model)   # shape: (batch_size, length, attention_output)

        return attention_output
    
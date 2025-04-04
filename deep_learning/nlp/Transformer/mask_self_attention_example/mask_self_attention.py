import numpy as np
import pandas as pd
import torch


# Check if we have GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the random seed
torch.manual_seed(42)

# simulation of word embedding with 512 dimensions (suppose that tokenization has been done.)
d_model = 5     # 512 / 768 / 1024
x1 = np.random.randn(d_model)     # I 
x2 = np.random.randn(d_model)     # like
x3 = np.random.randn(d_model)     # playing
x4 = np.random.randn(d_model)     # basketball
x5 = np.random.randn(d_model)     # .

# data
input_matrix = pd.DataFrame({"I": x1, "like": x2, "playing": x3, "basketball": x4, ".": x5}).T    # shape:(5, 512)
input_matrix = torch.from_numpy(input_matrix.values).to(dtype=torch.float32, device=device) # input_matrix.values is a ndarray.

# Create trainable weight matrix W_Q, W_K, and W_V, respectively
W_Q = torch.randn(d_model, d_model, requires_grad=True, device=device)
W_K = torch.randn(d_model, d_model, requires_grad=True, device=device)
W_V = torch.randn(d_model, d_model, requires_grad=True, device=device)

# We can compute the Query matrix, the Key matrix, and the Value matrix, respectively.
Q = input_matrix @ W_Q
K = input_matrix @ W_K
V = input_matrix @ W_V

# Compute Q @ K.T
QKT = Q @ K.T

#########################################################
#                   Mask Process                        #
#########################################################

# Create a mask matrix
mask = torch.tril(torch.ones(QKT.shape))

# Set mask format and replace 0 with -inf
QKT_mask = QKT.masked_fill(mask==0, -torch.inf)

# Compute attention scores
QKT_mask_attention = torch.softmax(QKT_mask, dim=-1)

# Compute attention outputs
output_mask_attention = QKT_mask_attention @ V

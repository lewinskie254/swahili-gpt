#%%
import torch 
import torch.nn as nn
from attention import SelfAttention_v2, MultiHeadAttentionWrapper
# %%

model = SelfAttention_v2(3, 6)

# %%
inputs = torch.rand((6, 3))

attention_weights = model(inputs)
# %%
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.8)
example = torch.ones(6, 3)
print(dropout(example))
# %%

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)

torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
batch.shape 
# %%
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

# %%
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],
                    [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]
                ]])
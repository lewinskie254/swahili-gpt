#%%
import torch 
import torch.nn as nn
from attention import SelfAttention_v2
# %%

model = SelfAttention_v2(6, 6)

# %%
inputs = torch.rand((6, 6))

attention_weights = model(inputs)
# %%
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.8)
example = torch.ones(6, 6)
print(dropout(example))
# %%
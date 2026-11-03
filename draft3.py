#%%
from gpt import TransformerBlock, GPT_CONFIG_124M
import torch 
import torch.nn  as nn 


#%%
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
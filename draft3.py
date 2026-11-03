#%%
from gpt import TransformerBlock, GPTModel, GPT_CONFIG_124M
import torch 
import torch.nn  as nn 


#%%
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
# %%
x[0, :1, :10]

#%%
output[0, :1, :10]
# %%

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)
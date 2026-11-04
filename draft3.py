#%%
from gpt import TransformerBlock, GPTModel, GPT_CONFIG_124M, generate_text_simple
import torch 
import torch.nn  as nn 
import tiktoken 

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
inputs = torch.tensor([
        [6109, 3626, 6100, 345],
        [6109, 1110, 6622, 257]
    ])

batch = inputs

print(inputs)
#%%
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)
# %%
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
# %%
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)
# %%
total_params_gpt2 = (total_params - sum(p.numel() for p in model.out_head.parameters()))
print(f"Number of trainable parameters "
f"considering weight tying: {total_params_gpt2:,}"
)
# %%
total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")
# %%

first_words = torch.tensor([[6109, 1110, 6622, 257]])

sentence = first_words
tokenizer = tiktoken.get_encoding("gpt2")
# %%
for i in range(20): 
    tokens = generate_text_simple(model, sentence, 3, 256)
    sentence = tokens 
    print(tokenizer.decode(sentence.squeeze().tolist()))


# %%

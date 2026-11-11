#%%
from gpt import TransformerBlock, GPTModel, GPT_CONFIG_124M, generate_text_simple, text_to_token_ids, token_ids_to_text
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

first_words = torch.tensor([[609, 111, 622, 257]])

sentence = first_words
tokenizer = tiktoken.get_encoding("gpt2")
# %%
start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(model=model,idx=text_to_token_ids(start_context, tokenizer),max_new_tokens=10,context_size=GPT_CONFIG_124M["context_length"])
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# %%
log_probas = torch.log(torch.cat((token_ids.squeeze()[0], token_ids.squeeze()[1])))
print(log_probas)
# %%
token_ids
# %%

sequences = torch.tensor([
    [6109, 3626, 6100, 345],
    [6109, 1110, 6622, 257]
])

inputs = sequences[:, :-1]
targets = sequences[:, 1:]

#%%
print(inputs.shape)   # (2, 4)
print(targets.shape)  # (2, 4)

print(token_ids_to_text(inputs[0], tokenizer))
print(token_ids_to_text(targets[0], tokenizer))
print(token_ids_to_text(inputs[1], tokenizer))
print(token_ids_to_text(targets[1], tokenizer))
#%%
with torch.no_grad():
    logits = model(inputs)

probas = torch.softmax(logits, dim=-1)

print(probas.shape)

#%%
text_idx = 0
target_probas_1 = probas[
    text_idx, [0, 1, 2], targets[text_idx]] 
print("Text 1:", target_probas_1.shape, target_probas_1)

#%%
print(targets[text_idx])
#%%
text_idx = 1
target_probas_2 = probas[
    text_idx, [0, 1, 2], targets[text_idx]
]
print("Text 2:", target_probas_2)
# %%
p1 = probas[0, 0, targets[0, 0]]
p2 = probas[0, 1, targets[0, 1]]
p3 = probas[0, 2, targets[0, 2]]

answers = token_ids_to_text(targets.flatten(), tokenizer)
list_of_answers = answers.strip().split(" ")
print(p1, targets[0, 0], list_of_answers[0])

print(p2, targets[0, 1], list_of_answers[1])

print(p3, targets[0, 2], list_of_answers[2])
# %%
targets[0, 0]
# %%
text_to_token_ids(answers, tokenizer)
# %%
torch.log(p1)
# %%
torch.log(torch.tensor(0.8))
# %%
torch.log(torch.tensor(0))
# %%
torch.log(torch.tensor(0.1))
# %%
torch.log(torch.tensor(0.0001))
# %%
p1
# %%
torch.log(torch.tensor(0.9))
# %%
print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)
# %%
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)
# %%
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)
# %%
logits_flat[:, [targets_flat]]
# %%
target_logits = logits_flat[
    torch.arange(targets_flat.shape[0]),
    targets_flat
]
# %%
target_logits
# %%
torch.sum(target_logits)
# %%
targets_flat
soft = torch.softmax(logits_flat, dim=-1)

# %%
populated = []
probabilities = []
for i, element in enumerate(targets_flat): 
    item = logits_flat[i, element]
    pro = soft[i, element]
    populated.append(item)
    probabilities.append(pro)
    print(i, element, item)
# %%
torch.log(torch.tensor(sum(populated)))
# %%
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)
# %%
import numpy as np 
populated_logs = [np.log(element) for element in populated]
sum(populated_logs)/len(populated)
# %%
populated_logs
# %%
populated
# %%
populated_soft = torch.softmax(torch.tensor(populated), dim=-1)
# %%
torch.softmax(torch.tensor([0.1, 0.8, 0.24, 0.9]), dim=-1)
# %%
torch.sum(populated_soft)
# %%
probs = torch.softmax(logits_flat, dim=-1)
# %%
x= probs[[
    torch.arange(targets_flat.shape[0]),
    targets_flat
]]
# %%
torch.sum(x)
# %%
y = logits_flat[[
    torch.arange(targets_flat.shape[0]),
    targets_flat
]]
# %%
y
# %%
sum(y)/targets_flat.shape[0]
# %%
torch.sum(torch.tensor(probabilities))
# %%
# Calculate log probabilities across the vocabulary
log_probs = torch.log_softmax(logits_flat, dim=-1)

# Select the correct target log-probability
target_log_probs = log_probs[
    torch.arange(targets_flat.shape[0]),
    targets_flat
]

# Negative mean log-likelihood
manual_loss = -target_log_probs.mean()

print("Manual loss:", manual_loss)
print("PyTorch loss:", loss)
# %%

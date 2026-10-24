#%%
import re 

#%%
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])
# %%
pattern = r'\W+'

words = re.split(pattern, raw_text)

# %%
len(words)
# %%
words 
# %%
unique_words = set(words)
print(len(unique_words))
# %%
unique_words
# %%
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
# %%
preprocessed
# %%
print(preprocessed[:30])
# %%
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)
# %%
print(all_words[:20])
# %%
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break
# %%
vocab
# %%
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}


    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


#%%
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
# %%
vocab["s"]
# %%
print(tokenizer.decode(ids))
# %%
text = "Hello, do you like tea?"
print(tokenizer.encode(text))
# %%
vocab["Hello"]
# %%
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab.items()))

# %%
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
# %%
tokenizer = SimpleTokenizerV2(vocab)
text = "Hello, do you like tea?"
print(tokenizer.encode(text))
# %%
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

# %%
print(tokenizer.encode(text))
# %%
from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))
# %%
tokenizer = tiktoken.get_encoding("gpt2")
# %%
text = (
"Hello, do you like tea? <|endoftext|> In the sunlit terraces"
"of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
# %%
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

# %%
enc_sample = enc_text[50:]
# %%
enc_sample
# %%
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
z = enc_sample[2:context_size+2]
print(f"x: {x}")
print(f"y: {y}")
print(f"z: {z}")
# %%
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
# %%


import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset): 
    def __init__(self, text, tokenizer, max_length, stride):
        super().__init__()
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride): 
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]



def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True,num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    num_workers=num_workers
                )
    return dataloader


#%%
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
# %%

dataloader = create_dataloader_v1(
raw_text, batch_size=8, max_length=4, stride=4,shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
# %%
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# %%
max_length = 4
dataloader = create_dataloader_v1(
raw_text, batch_size=8, max_length=max_length,
stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

#%%
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# %%
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)
# %%
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

# %%
token_embeddings[0].shape 

# %%
inputs = torch.tensor(
            [
                [0.43, 0.15, 0.89], # Your (x^1)
                [0.55, 0.87, 0.66], # journey (x^2)
                [0.57, 0.85, 0.64], # starts (x^3)
                [0.22, 0.58, 0.33], # with (x^4)
                [0.77, 0.25, 0.10], # one (x^5)
                [0.05, 0.80, 0.55] # step (x^6)
            ] 
)
# %%
inputs.shape
# %%
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)
# %%
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())
# %%
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)
attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# %%
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())
# %%
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
print(context_vec_2)
# %%
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)
print(attn_scores.shape)
# %%
attn_scores = inputs @ inputs.T #this is the same as the loop above. 
print(attn_scores)
print(attn_scores.shape, inputs.shape)
# %%
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
# %%
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
# %%
print("Previous 2nd context vector:", context_vec_2)
# %%
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2
# %%
x_2
# %%
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
# %%
W_query
# %%
W_key
# %%
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)
# %%
keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
# %%
keys
# %%
key_2 

# %%
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

# %%
query_2
# %%
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)
# %%
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)
# %%
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)
# %%
import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
# %%
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))
# %%
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec

#%%
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

# %%

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

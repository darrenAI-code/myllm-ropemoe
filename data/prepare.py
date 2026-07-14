"""
Prepare the Shakespeare dataset for language modeling with the o200k_base
tokenizer (the BPE tokenizer used by GPT-4o / o200k models).

Previously this script did character-level encoding. It now encodes with
tiktoken's o200k_base. Because o200k_base has ~200k tokens, the ids no longer
fit in uint16, so train.bin / val.bin are stored as uint32. meta.pkl records
the tokenizer name and dtype so that train.py / sample.py can pick them up.
"""
import os
import pickle
import requests
import numpy as np
import tiktoken

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# encode with the o200k_base BPE tokenizer
enc = tiktoken.get_encoding("o200k_base")
# o200k_base has max_token_value == 200018 (n_vocab == 200019). We pad the
# vocab size up to the nearest multiple of 64 for GPU efficiency, matching the
# convention used elsewhere in nanoGPT (50257 -> 50304).
vocab_size = ((enc.max_token_value + 1 + 63) // 64) * 64  # 200019 -> 200064
print(f"tokenizer: o200k_base, n_vocab: {enc.max_token_value + 1:,}, padded vocab_size: {vocab_size:,}")

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both splits to integers (encode_ordinary ignores any special tokens)
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files. uint32 is required since ids can exceed 2**16.
train_ids = np.array(train_ids, dtype=np.uint32)
val_ids = np.array(val_ids, dtype=np.uint32)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later.
# we no longer store stoi/itos; instead we record the tokenizer name and the
# on-disk dtype of the .bin files so the rest of the pipeline can adapt.
meta = {
    'vocab_size': vocab_size,
    'tokenizer': 'o200k_base',
    'dtype': 'uint32',
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# (token counts will be roughly 1/4 of the character count with BPE)

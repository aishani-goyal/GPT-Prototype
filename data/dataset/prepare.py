import os
import pickle
import numpy as np

# Path to your Hindi input file
input_file_path = os.path.join(os.path.dirname(__file__), r"C:\Users\AISHANI GOYAL\Downloads\GPT_Prototype\data\dataset\input.txt")

# Read Hindi data (make sure the file is saved in UTF-8 encoding)
with open(input_file_path, "r", encoding="utf-8") as f:
    data = f.read()

print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars[:200]))  # print only first 200 to avoid overload
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]  # encoder: string → list of ints

def decode(l):
    return ''.join([itos[i] for i in l])  # decoder: list of ints → string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files (compressed int16)
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

# save the meta information as well
meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("✅ Data preparation complete. Files saved: train.bin, val.bin, meta.pkl")
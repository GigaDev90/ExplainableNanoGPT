"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import numpy as np
import random
import itertools

def contains_no_two_consecutive_chars(new_str, existing_strings):
    for s in existing_strings:
        # Generate all substrings of length 3 from each string in existing_strings
        substrings = [s[i:i+2] for i in range(len(s) - 1)]
        
        # Check if any of these substrings appear in new_str
        if any(sub in new_str for sub in substrings):
            return False  # Found a forbidden substring in new_str
    return True  # No forbidden substrings found in new_str

def generate_unique_string(length, char_set, existing_strings, unique):
    if not unique:
        new_str = ''.join(random.choices(char_set, k=length))
        return new_str
    while True:
        new_str = ''.join(random.choices(char_set, k=length))
        if contains_no_two_consecutive_chars(new_str, existing_strings):
            return new_str

def create_random_text(strings, text_length, space):
    if space:
      return ' '.join(random.choices(list(strings), k=text_length))
    else:
      return ''.join(random.choices(list(strings), k=text_length))

def generate_synthetic_dataset(length, num_strings, text_length, filename, unique=True, space=True):
    char_set = 'abcdefghijklmnopqrstuvwxyz'  # Define your character set
    existing_strings = set()

    # Generate unique strings
    while len(existing_strings) < num_strings:
        unique_str = generate_unique_string(length, char_set, existing_strings, unique)
        existing_strings.add(unique_str)

    # Create random text from unique strings
    random_text = create_random_text(existing_strings, text_length, space)

    # Save the generated text to a file
    with open(os.path.join(dataset, filename), 'w') as file:
        file.write(random_text)

    return random_text


token_length = 5
token_number = 100
text_length = 50000
uniqueness = True
spaced = True
input_file_path = "input.txt"
dataset = "synthetic_data_char_default"

exec(open('configurator.py').read()) # overrides from command line or config file
dataset = os.path.join('data', dataset)

os.makedirs(dataset, exist_ok=True)

generate_synthetic_dataset(token_length, token_number, text_length, input_file_path, uniqueness, spaced)

with open(os.path.join(dataset, input_file_path), 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(dataset, 'train.bin'))
val_ids.tofile(os.path.join(dataset, 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(dataset, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens

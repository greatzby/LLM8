import os
import pickle
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser(description='Create the dataset based on the given parameters.')  
parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes in the graph')  
parser.add_argument('--num_of_paths', type=int, default=20, help='Number of paths per pair nodes in training dataset')  
args = parser.parse_args()  

num_nodes = args.num_nodes

if(args.num_of_paths == 0):
    train_file_path = os.path.join(os.path.dirname(__file__), f'{args.num_nodes}/train.txt')
    val_file_path = os.path.join(os.path.dirname(__file__), f'{args.num_nodes}/test.txt')
else:
    train_file_path = os.path.join(os.path.dirname(__file__), f'{args.num_nodes}/train_{args.num_of_paths}.txt')
    val_file_path = os.path.join(os.path.dirname(__file__), f'{args.num_nodes}/test.txt')
# test_file_path = os.path.join(os.path.dirname(__file__), 'test.txt')

with open(train_file_path, 'r') as f:
    train_data = f.read()
print(f"length of train dataset in characters: {len(train_data):,}")

with open(val_file_path, 'r') as f:
    val_data = f.read()
print(f"length of val dataset in characters: {len(val_data):,}")

all_data = train_data + val_data

def find_characters(data_string):
    pattern = r'\d+|\D'
    matches = re.findall(pattern, data_string)
    return set(matches)

def process_reasoning(s):
    split_text = s.split('\n')
    #split_text = [s + '\n' for s in split_text if s != ""]
    ret = []
    for st in split_text:
        if(st != ""):
            enc_str = encode(st) + [1]
            ret += enc_str +[0] * (block_size + 1 - len(enc_str))
    return ret

def get_block_size(s):
    split_text = s.split('\n')
    #split_text = [s + '\n' for s in split_text if s != ""]
    ret = []
    bs = 0
    for st in split_text:
        if(st != ""):
            enc_str = encode(st) + [1]
            bs = max(bs, len(enc_str))
    return bs


def encode_string(s, stonum):
    ss = s.split(" ")
    encoded_string = [stonum[ch] for ch in ss]
    return encoded_string

def decode_string(l, numtos):
    dec = ""
    for i in l:
        dec = dec + numtos[i] + " "
    return dec[:-1]


# get all the unique characters that occur in this text
chars = sorted(list(find_characters(all_data)))
vocab_size = num_nodes+2
print("all the unique characters:", ' '.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = {}
itos = {}

for i in range(num_nodes):
    stoi[str(i)] = i+2
    itos[i+2] = str(i)

stoi['[PAD]'] = 0
itos[0] = '[PAD]'
stoi['\n'] = 1
itos[1] = '\n'

def encode(s):
    return encode_string(s, stoi) # encoder: take a string, output a list of integers
def decode(l):
    return decode_string(l, itos) # decoder: take a list of integers, output a string

# encode both to integers
block_size = (max(get_block_size(train_data), get_block_size(val_data)) // 32 + 1) * 32

print(f"the block size is {block_size}")

train_ids = process_reasoning(train_data)

val_ids = process_reasoning(val_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

if(args.num_of_paths == 0):
    train_ids.tofile(os.path.join(os.path.dirname(__file__), f'{args.num_nodes}/train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), f'{args.num_nodes}/val.bin'))
else:
    train_ids.tofile(os.path.join(os.path.dirname(__file__), f'{args.num_nodes}/train_{args.num_of_paths}.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), f'{args.num_nodes}/val.bin'))


unreachable = False; simple_format = True
if 'x' in chars:
    unreachable = True
if ':' in chars:
    simple_format = False
    

# save the meta information as well, to help us encode/decode later
meta = {
    'unreachable': unreachable,
    'simple_format': simple_format,
    'block_size': block_size,
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

print(stoi)
print(itos)
with open(os.path.join(os.path.dirname(__file__), f'{args.num_nodes}/meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
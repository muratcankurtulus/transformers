import torch
import joblib

from model import Transformer
from tokenization import encode, decode

# load merges and vocab
merges = joblib.load('./toy_data/merges')
vocab = joblib.load('./toy_data/vocab')
# Load the model
state_dict = torch.load('./transformer_epoch_5.pth')

model = Transformer(embed_dim=512,
                    src_vocab_size=1024,
                    tgt_vocab_size=1024,
                    seq_len=10,
                    num_layers=4,
                    expansion_factor=4,
                    n_heads=8).to("cuda")
model.load_state_dict(state_dict)
model.eval()

string = "Hello, my name is"
tokens = encode(string, merges)
src_tokens = torch.tensor(tokens).unsqueeze(0).to("cuda")
tgt_tokens = torch.tensor(tokens[-1:]).unsqueeze(0).to("cuda")
output = model(src_tokens, tgt_tokens)
print(f"output shae: {output.shape}")
print(f"output: {output}")
output = output.view(-1, 1024)
print(f"output shae: {output.shape}")
print(f"output: {output}")
output = torch.argmax(output, dim=-1)
print(f"output shae: {output.shape}")
print(f"output: {output}")
output = decode(output, vocab)
print(output)

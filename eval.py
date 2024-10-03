import torch
from model import GPT
from pydantic import BaseModel

from tokenization import Tokenizer


class ModelConfig(BaseModel):
    embed_dim: int = 512
    tgt_vocab_size: int = 1024
    seq_len: int = 34
    num_layers: int = 4
    expansion_factor: int = 4
    n_heads: int = 8


state_dict = torch.load('./transformer_epoch_95.pth')
model_config = ModelConfig()
model = GPT(**model_config.dict()).to("cuda")
tokenizer = Tokenizer.load("toy_data/test")
model.load_state_dict(state_dict)
model.eval()

string = "it's a "
input_ids = tokenizer.encode(string)
generated = model.generate(input_ids, 20)
generated = generated.cpu().tolist()
print(tokenizer.decode(generated[0]))

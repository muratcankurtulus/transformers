import torch
from pydantic import BaseModel

from gpt import GPT
from tokenization import Tokenizer


class ModelConfig(BaseModel):
    embed_dim: int = 32
    tgt_vocab_size: int = 512
    seq_len: int = 256
    num_layers: int = 2
    expansion_factor: int = 2
    n_heads: int = 2


state_dict = torch.load('./gpt_epoch_80.pth')
model_config = ModelConfig()
model = GPT(**model_config.dict()).to("cuda")
tokenizer = Tokenizer.load("toy_data/python_book")
model.load_state_dict(state_dict)
model.eval()

string = "python is "
input_ids = tokenizer.encode(string)
generated = model.generate(input_ids, 200)
generated = generated.cpu().tolist()
print(tokenizer.decode(generated[0]))

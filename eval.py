import torch
from pydantic import BaseModel

from gpt import GPT
from tokenization import Tokenizer


class ModelConfig(BaseModel):
    embed_dim: int = 512
    tgt_vocab_size: int = 4096
    seq_len: int = 256
    num_layers: int = 6
    expansion_factor: int = 6
    n_heads: int = 8


state_dict = torch.load('./transformer_epoch_25.pth')
model_config = ModelConfig()
model = GPT(**model_config.dict()).to("cuda")
tokenizer = Tokenizer.load("toy_data/python_book")
model.load_state_dict(state_dict)
model.eval()

string = "def "
input_ids = tokenizer.encode(string)
generated = model.generate(input_ids, 200)
generated = generated.cpu().tolist()
print(tokenizer.decode(generated[0]))

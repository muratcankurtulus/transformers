import torch
import torch.nn as nn
import torch.optim as optim
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


class DatasetConfig(BaseModel):
    batch_size: int = 16
    shuffle: bool = True


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data, seq_len, tokenizer):
        self.seq_len = seq_len - 2
        self.tokenizer = tokenizer
        self.data = self.tokenizer.encode(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx + self.seq_len + 1 > len(self.data):
            idx = 0
        src = self.tokenizer.add_special_tokens(self.data[idx:idx +
                                                          self.seq_len])
        tgt = self.tokenizer.add_special_tokens(self.data[idx + 1:idx +
                                                          self.seq_len + 1])
        return torch.tensor(src).to("cuda"), torch.tensor(tgt).to("cuda")


def main():
    # Load tokenizer
    tokenizer = Tokenizer.load("toy_data/test")

    # Model configuration
    model_config = ModelConfig()
    model = GPT(**model_config.dict()).to("cuda")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=0.0001,
                           betas=(0.9, 0.98),
                           eps=1e-9)

    # Load data and create DataLoader
    with open("toy_data/example.txt", encoding="utf-8") as f:
        data = f.read()

    dataset_config = DatasetConfig()
    train_loader = torch.utils.data.DataLoader(
        Dataset(data, model_config.seq_len, tokenizer),
        batch_size=dataset_config.batch_size,
        shuffle=dataset_config.shuffle)

    # Training loop
    for epoch in range(100):
        model.train()
        for i, (src, tgt) in enumerate(train_loader):
            mask = model.make_tgt_mask(tgt).to("cuda")
            optimizer.zero_grad()
            output = model(src, mask)
            loss = criterion(output.view(-1, 1024), tgt.view(-1))
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'transformer_epoch_{epoch}.pth')
            print('Model saved!')


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import joblib

from model import Transformer

#defining the model parameters in detail
model = Transformer(embed_dim=512,
                    src_vocab_size=1024,
                    tgt_vocab_size=1024,
                    seq_len=10,
                    num_layers=4,
                    expansion_factor=4,
                    n_heads=8).to("cuda")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


#creating the dataset
class Dataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx + 20 > len(self.data):
            idx = 0
        src = self.data[idx:idx + 10]
        tgt = self.data[idx + 10:idx + 20]
        return torch.tensor(src).to("cuda"), torch.tensor(tgt).to("cuda")


#creating the train dataloader
data = joblib.load('toy_data/ids')
train_loader = torch.utils.data.DataLoader(Dataset(data), batch_size=8, shuffle=True)

#training the model
for epoch in range(10):
    for i, (src, tgt) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(src, tgt)
        print(output.view(-1, 1024))
        loss = criterion(output.view(-1, 1024), tgt.view(-1))
        loss.backward()
        optimizer.step()
        break
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'transformer_epoch_{epoch}.pth')
        print('Model saved!')

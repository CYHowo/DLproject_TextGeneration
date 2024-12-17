import torch
import wandb
from datetime import datetime
import requests
import torch.nn as nn
from pathlib import Path
from model import CharDataset, TransformerConfig, DecoderOnlyTransformer
from torch.utils.data import DataLoader

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

## Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Download the data
path_to_file = "shakesperean.txt"
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

## Check if file exists
if not Path(path_to_file).exists():
    with open(path_to_file, "w", encoding='utf-8') as file:
        file.write(requests.get(url).text)

data=open(path_to_file,'rb').read().decode(encoding='utf-8')
# print("length of text:{} characters".format(len(data)))
vocab = sorted(set(data))
# print('unique characters:{}'.format(len(vocab)))


## Init Dataset and set the config
config = TransformerConfig(vocab_size=128, n_layers=12, n_heads=8, embed_dim=768, block_size=128, dropout=0.1)
dataset = CharDataset(config, data)

## Initialize wandb
wandb.init(project="shakesperean", config=config)

model = DecoderOnlyTransformer(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
print(model)

num_epochs = 5
batch_size = 64
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

## Train the model
model.train()
i=0
for epoch in range(num_epochs):
    total_loss = 0
    # the way not using DataLoader
    # for i in range(0, len(dataset) - batch_size, batch_size):
    #    input_data, target_data = zip(*[dataset[j] for j in range(i, i + batch_size)])
    for input_data, target_data in train_loader:
        
        input_data = input_data.to(device)  # (batch_size, seq_len)
        target_data = target_data.to(device)  # (batch_size, seq_len)

        optimizer.zero_grad()
        output_logits = model(input_data)  # (batch_size, seq_len, vocab_size)
        batch_size, seq_len, vocab_size = output_logits.shape
        loss = criterion(output_logits.reshape(-1, vocab_size), target_data.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (len(dataset) // batch_size)
    wandb.log({"epoch": epoch + 1, "loss": avg_loss})
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')

    torch.save(model.state_dict(), f"model/best_model_{current_time}.pth")

# model.load_state_dict(torch.load("model/best_model.pth", map_location=device))

## Generate text
model.eval()
context = "O God, O God!"

# Tokenize the context
tokenized_context = torch.tensor([dataset.stoi[ch] for ch in context], dtype=torch.long).unsqueeze(0).to(device)
# Generate text
generated = model.generate(tokenized_context, max_length=100)
# Ensure generated tokens are within the valid range
generated = torch.clamp(generated, min=0, max=len(dataset.itos)-1)
# Convert generated tokens back to string using inverse mapping.
generated_text = ''.join([dataset.itos[i.item()] for i in generated.squeeze()])
print(generated_text)
wandb.log({"generated_text": generated_text})




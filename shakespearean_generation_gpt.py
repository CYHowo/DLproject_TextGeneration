import torch
import wandb
import requests
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")  # 格式化為 "年-月-日_時-分-秒"

## Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Download the data
path_to_file = "shakesperean.txt"
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if not Path(path_to_file).exists():
    with open(path_to_file, "w", encoding='utf-8') as file:
        file.write(requests.get(url).text)

data = open(path_to_file, 'r', encoding='utf-8').read()

## Load GPT tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

## Prepare Dataset
class GPTDataset(Dataset):
    def __init__(self, data, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.inputs = self.tokenizer(
            data,
            return_tensors="pt",
            truncation=True,
            max_length=self.block_size,
            padding="max_length",
        )["input_ids"]

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        input_ids = self.inputs[idx]
        target_ids = input_ids.clone()
        return input_ids, target_ids

block_size = 128
dataset = GPTDataset(data, tokenizer, block_size)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

## Configure WandB
wandb.init(project="shakesperean", config={"model": "gpt2", "block_size": block_size})

optimizer = AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

## Training Loop
model.train()
num_epochs = 1000
best_loss = float("inf")

for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        input_ids, target_ids = batch
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    wandb.log({"epoch": epoch + 1, "loss": avg_loss})
    # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

    # Save the best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), f"model/best_model_{current_time}.pth")

wandb.finish()

## Text Generation
model.eval()
context = "O God, O God!"
input_ids = tokenizer(context, return_tensors="pt")["input_ids"].to(device)
generated_ids = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

wandb.log({"generated_text": generated_text})

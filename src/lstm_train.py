import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import os

from data_utils import process_file_and_save, tokens_to_indices, pad_sequences_torch
from next_token_dataset import TextDataset
from lstm_model import LSTMModel
from eval_lstm import train_epoch, eval_epoch
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

input_file = 'data/tweets.txt'

samples = process_file_and_save(input_file)
all_tokens = [token for sample in samples for token in sample['X']] + [token for sample in samples for token in sample['Y']]
vocab = {token: idx+1 for idx, token in enumerate(sorted(set(all_tokens)))}
vocab_size = len(vocab) + 1

X_indices, Y_indices = tokens_to_indices(samples, vocab)
max_len = max(len(x) for x in X_indices)
X_pad = pad_sequences_torch(X_indices, max_len, torch)
Y_pad = pad_sequences_torch(Y_indices, max_len, torch)

X_train, X_temp, Y_train, Y_temp = train_test_split(X_pad, Y_pad, test_size=0.2, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

os.makedirs('data', exist_ok=True)

def tensor_to_str_list(tensor):
    return [' '.join(map(str, seq.tolist())) for seq in tensor]

pd.DataFrame({'X': tensor_to_str_list(X_train), 'Y': tensor_to_str_list(Y_train)}).to_csv('/home/assistant/text-autocomplete/data/train.csv', index=False, encoding='utf-8')
pd.DataFrame({'X': tensor_to_str_list(X_val),   'Y': tensor_to_str_list(Y_val)}).to_csv('/home/assistant/text-autocomplete/data/val.csv', index=False, encoding='utf-8')
pd.DataFrame({'X': tensor_to_str_list(X_test),  'Y': tensor_to_str_list(Y_test)}).to_csv('/home/assistant/text-autocomplete/data/test.csv', index=False, encoding='utf-8')

train_dataset = TextDataset(X_train, Y_train)
val_dataset = TextDataset(X_val, Y_val)
test_dataset = TextDataset(X_test, Y_test)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

embedding_dim = 128
hidden_dim = 256
model = LSTMModel(vocab_size, embedding_dim, hidden_dim).to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters())

train_losses = []
val_losses = []

epochs = 1
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = eval_epoch(model, val_loader, criterion, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

torch.save(model.state_dict(), "/home/assistant/text-autocomplete/models/model_lstm_weights.pth")
print("Model weights saved to models/model_lstm_weights.pth")

plt.figure(figsize=(10,6))
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

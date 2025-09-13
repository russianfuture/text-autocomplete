import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        out = self.fc(lstm_out)
        return out

    def generate(self, tokenizer, vocab, seed_text, max_length=20):
        self.eval()
        tokens = tokenizer.tokenize(seed_text.lower())
        with torch.no_grad():
            for _ in range(max_length):
                x_idx = [vocab.get(token, 0) for token in tokens]
                x_tensor = torch.tensor([x_idx], dtype=torch.long).to(next(self.parameters()).device)
                output = self.forward(x_tensor)
                last_logits = output[0, len(tokens) - 1]
                next_id = torch.argmax(last_logits).item()
                next_token = None
                for tok, idx in vocab.items():
                    if idx == next_id:
                        next_token = tok
                        break
                if next_token is None or next_token == '[SEP]':
                    break
                tokens.append(next_token)
                if len(tokens) >= max_length:
                    break
        return ' '.join(tokens)

import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, mode='train', max_len=50, start_token=None):
        if mode == 'train':
            emb = self.embedding(x)
            out, _ = self.lstm(emb)
            logits = self.fc(out)
            return logits
        elif mode == 'generate':
            # Генерация последовательности по одному токену
            assert start_token is not None, "start_token should be provided for generation"
            generated = [start_token]
            h, c = None, None
            for _ in range(max_len):
                input_token = torch.tensor([[generated[-1]]], device=x.device)
                emb = self.embedding(input_token)
                out, (h, c) = self.lstm(emb, (h, c)) if h is not None else self.lstm(emb)
                logits = self.fc(out[:, -1, :])
                next_token = torch.argmax(logits, dim=-1).item()
                generated.append(next_token)
            return generated

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

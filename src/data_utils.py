import re
import pandas as pd
import os
from transformers import AutoTokenizer
from rouge_score import rouge_scorer

tokenizer_lstm = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\sа-яё]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_training_samples(tokens):
    if len(tokens) < 2:
        return None
    return tokens[:-1], tokens[1:]

def process_file_and_save(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()

    clean_texts = [clean_text(t) for t in texts if t.strip()]
    os.makedirs('data', exist_ok=True)
    df_raw = pd.DataFrame({'text': clean_texts})
    df_raw.to_csv('/home/assistant/text-autocomplete/data/raw_dataset.csv', index=False, encoding='utf-8')
    print("Raw dataset saved to data/raw_dataset.csv")

    samples = []
    for text in clean_texts:
        tokens = tokenizer_lstm.tokenize(text)
        pair = create_training_samples(tokens)
        if pair:
            X, Y = pair
            samples.append({'X': X, 'Y': Y})

    df_token = pd.DataFrame(samples)
    df_token['X_str'] = df_token['X'].apply(lambda x: ' '.join(x))
    df_token['Y_str'] = df_token['Y'].apply(lambda x: ' '.join(x))
    df_token[['X_str','Y_str']].to_csv('/home/assistant/text-autocomplete/data/dataset_processed.csv', index=False, encoding='utf-8')
    print("Tokenized dataset saved to data/dataset_processed.csv")
    return samples

def tokens_to_indices(samples, vocab):
    X_indices, Y_indices = [], []
    for sample in samples:
        x_idx = [vocab.get(token, 0) for token in sample['X']]
        y_idx = [vocab.get(token, 0) for token in sample['Y']]
        X_indices.append(x_idx)
        Y_indices.append(y_idx)
    return X_indices, Y_indices

def pad_sequences_torch(sequences, max_len, torch):
    padded = []
    for seq in sequences:
        s = seq[:max_len]
        padded_seq = s + [0]*(max_len - len(s))
        padded.append(torch.tensor(padded_seq, dtype=torch.long))
    return torch.stack(padded)

def compute_rouge(reference, prediction):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure

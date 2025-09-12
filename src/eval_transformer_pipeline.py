from transformers import pipeline, AutoTokenizer
import re
import numpy as np
from data_utils import compute_rouge

generator_distilgpt2 = pipeline("text-generation", model="distilgpt2")
tokenizer_distilgpt2 = AutoTokenizer.from_pretrained("distilgpt2")

def split_text_for_completion(text):
    tokens = tokenizer_distilgpt2.tokenize(text)
    cut_off = (len(tokens) * 3) // 4
    input_text = tokenizer_distilgpt2.convert_tokens_to_string(tokens[:cut_off])
    target_text = tokenizer_distilgpt2.convert_tokens_to_string(tokens[cut_off:])
    return input_text, target_text

def clean_text_for_distilgpt2(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\sа-яёa-z]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

input_file = 'data/tweets.txt'
texts = [clean_text_for_distilgpt2(t) for t in open(input_file, encoding='utf-8').readlines() if len(t.strip()) > 10]
_, val_texts = train_test_split(texts, test_size=0.2, random_state=42)

rouge1_lstm_scores, rouge2_lstm_scores = [], []
rouge1_gpt2_scores, rouge2_gpt2_scores = [], []

print("Evaluating transformer distilgpt2 model...")

for idx, text in enumerate(val_texts[:50]):
    input_text, ref_text = split_text_for_completion(text)

    gpt2_out = generator_distilgpt2(input_text,
                                    max_length=len(tokenizer_distilgpt2.encode(input_text + ref_text)),
                                    do_sample=True, top_k=50, num_return_sequences=1)
    gpt2_pred = gpt2_out[0]['generated_text'][len(input_text):].strip()
    r1_gpt2, r2_gpt2 = compute_rouge(ref_text, gpt2_pred)
    rouge1_gpt2_scores.append(r1_gpt2)
    rouge2_gpt2_scores.append(r2_gpt2)

    if idx < 5:
        print(f"\nExample #{idx+1}")
        print("Input:", input_text)
        print("Reference:", ref_text)
        print(f"distilgpt2 Prediction: {gpt2_pred}")
        print(f"ROUGE-1: {r1_gpt2:.3f}, ROUGE-2: {r2_gpt2:.3f}")

print(f"\nAverage distilgpt2 ROUGE-1: {np.mean(rouge1_gpt2_scores):.3f}")
print(f"Average distilgpt2 ROUGE-2: {np.mean(rouge2_gpt2_scores):.3f}")

import json
import os
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

JAILBREAK_PATH = "./data/generated/crescendo_all.jsonl"
BENIGN_PATH = "./data/generated/benign_topic_matched.jsonl"

def load_data(path):
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def get_user_text(conv):
    return " ".join(t["content"] for t in conv["turns"] if t["role"] == "user")

def main():
    print("Loading data...", flush=True)
    jailbreak = load_data(JAILBREAK_PATH)
    benign = load_data(BENIGN_PATH)
    print(f"Jailbreak: {len(jailbreak)}, Benign: {len(benign)}", flush=True)

    print("\n=== Check 1: DeBERTa embeddings + LogReg ===", flush=True)
    print("Loading DeBERTa model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModel.from_pretrained("microsoft/deberta-v3-base", torch_dtype=torch.float32).eval().cuda()
    print("DeBERTa loaded!", flush=True)

    def encode_conv(conv):
        user_turns = [t["content"] for t in conv["turns"] if t["role"] == "user"]
        text = " [SEP] ".join(user_turns)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            out = model(**inputs)
        return out.last_hidden_state[:, 0, :].cpu().numpy().flatten()

    print("Encoding jailbreak conversations...", flush=True)
    X_j = np.array([encode_conv(c) for c in jailbreak])
    print("Encoding benign conversations...", flush=True)
    X_b = np.array([encode_conv(c) for c in benign])

    X = np.vstack([X_j, X_b])
    y = np.array([1]*len(X_j) + [0]*len(X_b))

    print("Running DeBERTa cross-validation...", flush=True)
    scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5, scoring="f1_macro")
    print(f"DeBERTa LogReg F1: {scores.mean():.3f} +/- {scores.std():.3f}", flush=True)

    del model
    torch.cuda.empty_cache()

    print("\n=== Check 2: TF-IDF only ===", flush=True)
    texts = [get_user_text(c) for c in jailbreak] + [get_user_text(c) for c in benign]
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(texts)
    scores_tfidf = cross_val_score(LogisticRegression(max_iter=1000), X_tfidf, y, cv=5, scoring="f1_macro")
    print(f"TF-IDF LogReg F1: {scores_tfidf.mean():.3f} +/- {scores_tfidf.std():.3f}", flush=True)

    print(f"\n{'='*50}", flush=True)
    print(f"Sanity Check Results:", flush=True)
    print(f"  DeBERTa F1: {scores.mean():.3f} (target: 0.70-0.85)", flush=True)
    print(f"  TF-IDF F1: {scores_tfidf.mean():.3f} (target: 0.50-0.65)", flush=True)
    deberta_pass = 0.70 <= scores.mean() <= 0.85
    tfidf_pass = scores_tfidf.mean() <= 0.65
    print(f"  DeBERTa PASS: {deberta_pass}", flush=True)
    print(f"  TF-IDF PASS: {tfidf_pass}", flush=True)
    print(f"{'='*50}", flush=True)

    lr = LogisticRegression(max_iter=1000).fit(X_tfidf, y)
    feature_names = tfidf.get_feature_names_out()
    top_positive = np.argsort(lr.coef_[0])[-20:][::-1]
    top_negative = np.argsort(lr.coef_[0])[:20]
    print("\nTop TF-IDF features (jailbreak-indicative):", flush=True)
    for i in top_positive:
        print(f"  {feature_names[i]}: {lr.coef_[0][i]:.3f}", flush=True)
    print("\nTop TF-IDF features (benign-indicative):", flush=True)
    for i in top_negative:
        print(f"  {feature_names[i]}: {lr.coef_[0][i]:.3f}", flush=True)

if __name__ == "__main__":
    main()

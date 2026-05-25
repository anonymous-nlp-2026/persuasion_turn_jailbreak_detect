import sys, json, torch, os
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, '.')
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier
from src.data.dataset import ConversationDataset

device = torch.device('cuda:0')
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

# Load treatment encoder
print("Loading treatment DeBERTa...")
model = DeBERTaMultiTask(model_name='microsoft/deberta-v3-base')
state = torch.load('checkpoints/plan_002/deberta_multitask/best/model.pt', map_location='cpu')
model.load_state_dict(state)
treatment_enc = model.deberta.to(device).eval()
for p in treatment_enc.parameters():
    p.requires_grad = False

# Load baseline encoder
print("Loading baseline DeBERTa...")
baseline_enc = AutoModel.from_pretrained('microsoft/deberta-v3-base').to(device).eval()
for p in baseline_enc.parameters():
    p.requires_grad = False

embed_dim = treatment_enc.config.hidden_size

# Precompute embeddings
def precompute_all(dataset, encoder, max_length=256):
    all_embs, all_labels, all_lengths = [], [], []
    for conv in dataset.conversations:
        turns = conv["turns"]
        if not turns:
            turns = [""]
        enc = tokenizer(turns, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = out.last_hidden_state[:, 0, :].cpu()
        all_embs.append(embs)
        all_labels.append(conv["label"])
        all_lengths.append(embs.size(0))
    return all_embs, all_labels, all_lengths

# Load datasets
train_ds = ConversationDataset('data/plan_002_splits/train.jsonl')
val_ds = ConversationDataset('data/plan_002_splits/val.jsonl')
test_ds = ConversationDataset('data/plan_002_splits/test.jsonl')
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

# Precompute all embeddings
print("Precomputing treatment embeddings...")
t_train_embs, t_train_labels, t_train_lens = precompute_all(train_ds, treatment_enc)
t_val_embs, t_val_labels, t_val_lens = precompute_all(val_ds, treatment_enc)
t_test_embs, t_test_labels, t_test_lens = precompute_all(test_ds, treatment_enc)

print("Precomputing baseline embeddings...")
b_train_embs, b_train_labels, b_train_lens = precompute_all(train_ds, baseline_enc)
b_val_embs, b_val_labels, b_val_lens = precompute_all(val_ds, baseline_enc)
b_test_embs, b_test_labels, b_test_lens = precompute_all(test_ds, baseline_enc)

# Free GPU memory from encoders
del treatment_enc, baseline_enc, model
torch.cuda.empty_cache()

# GRU training function
import torch.nn as nn

def pad_sequences(embs_list, labels_list, lengths_list):
    max_len = max(lengths_list)
    padded = torch.zeros(len(embs_list), max_len, embs_list[0].size(1))
    for i, e in enumerate(embs_list):
        padded[i, :e.size(0), :] = e
    labels = torch.tensor(labels_list, dtype=torch.long)
    lengths = torch.tensor(lengths_list, dtype=torch.long)
    return padded, labels, lengths

def train_gru(train_embs, train_labels, train_lens, val_embs, val_labels, val_lens, name, epochs=20, lr=1e-3):
    print(f"\n--- Training GRU ({name}) ---")
    gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(gru.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_padded, train_y, train_l = pad_sequences(train_embs, train_labels, train_lens)
    val_padded, val_y, val_l = pad_sequences(val_embs, val_labels, val_lens)
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        gru.train()
        logits = gru(train_padded.to(device), train_l.to(device))
        loss = criterion(logits, train_y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        gru.eval()
        with torch.no_grad():
            val_logits = gru(val_padded.to(device), val_l.to(device))
            val_loss = criterion(val_logits, val_y.to(device)).item()
            val_acc = (val_logits.argmax(-1).cpu() == val_y).float().mean().item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in gru.state_dict().items()}
            print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} *")
        elif (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    # Save checkpoint
    ckpt_dir = Path(f'checkpoints/plan_002/gru/{name}')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_dir / 'best.pt')
    print(f"  Saved to {ckpt_dir / 'best.pt'}")
    
    gru.load_state_dict(best_state)
    return gru

# Train both GRUs
treatment_gru = train_gru(t_train_embs, t_train_labels, t_train_lens, t_val_embs, t_val_labels, t_val_lens, "treatment")
baseline_gru = train_gru(b_train_embs, b_train_labels, b_train_lens, b_val_embs, b_val_labels, b_val_lens, "baseline")

# Evaluation
def eval_gru_at_k(gru, test_embs, test_labels, test_lens, k):
    all_embs, all_labels, all_lengths = [], [], []
    for embs, label, length in zip(test_embs, test_labels, test_lens):
        if k is not None:
            trunc = min(k, embs.size(0))
            embs = embs[:trunc]
        all_embs.append(embs)
        all_labels.append(label)
        all_lengths.append(embs.size(0))
    
    padded, labels_t, lengths_t = pad_sequences(all_embs, all_labels, all_lengths)
    
    gru.eval()
    with torch.no_grad():
        logits = gru(padded.to(device), lengths_t.to(device))
        preds = logits.argmax(-1).cpu().numpy()
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    
    labels_np = labels_t.numpy()
    
    # FPR@95TPR
    benign_p = probs[labels_np == 0]
    jb_p = probs[labels_np == 1]
    if len(jb_p) > 0 and len(benign_p) > 0:
        th = np.sort(jb_p)
        idx = max(0, int(np.floor(len(th) * 0.05)) - 1)
        fpr = float((benign_p >= th[idx]).mean())
    else:
        fpr = 0.0
    
    return {
        'f1': float(f1_score(labels_np, preds, zero_division=0)),
        'precision': float(precision_score(labels_np, preds, zero_division=0)),
        'recall': float(recall_score(labels_np, preds, zero_division=0)),
        'accuracy': float(accuracy_score(labels_np, preds)),
        'fpr_at_95tpr': fpr
    }

# TF-IDF evaluation
def load_raw_convs(path):
    convs = []
    with open(path) as f:
        for line in f:
            c = json.loads(line.strip())
            turns = [t['content'] for t in c['turns'] if t['role'] == 'user']
            label = 1 if c['label'] == 'jailbreak' else 0
            convs.append({'turns': turns, 'label': label})
    return convs

train_raw = load_raw_convs('data/plan_002_splits/train.jsonl')
test_raw = load_raw_convs('data/plan_002_splits/test.jsonl')

def eval_tfidf_at_k(train_raw, test_raw, k):
    def make_texts(convs, k):
        texts, labels = [], []
        for c in convs:
            turns = c['turns'][:k] if k is not None else c['turns']
            texts.append(' '.join(turns) if turns else '')
            labels.append(c['label'])
        return texts, labels
    
    tr_texts, tr_labels = make_texts(train_raw, k)
    te_texts, te_labels = make_texts(test_raw, k)
    
    vec = TfidfVectorizer(max_features=5000)
    X_tr = vec.fit_transform(tr_texts)
    X_te = vec.transform(te_texts)
    
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_tr, tr_labels)
    preds = lr.predict(X_te)
    
    return {
        'f1': float(f1_score(te_labels, preds, zero_division=0)),
        'precision': float(precision_score(te_labels, preds, zero_division=0)),
        'recall': float(recall_score(te_labels, preds, zero_division=0)),
        'accuracy': float(accuracy_score(te_labels, preds)),
        'fpr_at_95tpr': 0.0
    }

# Run all evaluations
print("\n=== Full Evaluation ===")
results = {"early_detection": {}, "stage1": {"best_epoch": 3, "note": "plan_002 DeBERTa multitask"}}
K_values = [1, 2, 3, 5, None]

for k in K_values:
    k_label = str(k) if k is not None else "full"
    print(f"\nK={k_label}:")
    
    t_res = eval_gru_at_k(treatment_gru, t_test_embs, t_test_labels, t_test_lens, k)
    b_res = eval_gru_at_k(baseline_gru, b_test_embs, b_test_labels, b_test_lens, k)
    tfidf_res = eval_tfidf_at_k(train_raw, test_raw, k)
    
    print(f"  Treatment F1={t_res['f1']:.4f} | Baseline F1={b_res['f1']:.4f} | TF-IDF F1={tfidf_res['f1']:.4f}")
    
    results["early_detection"][k_label] = {
        "treatment": t_res,
        "baseline": b_res,
        "tfidf": tfidf_res
    }

# Print table
print("\n" + "=" * 80)
print(f"{'plan_002 Evaluation Results':^80}")
print("=" * 80)
print(f"\n{'K':<6} {'Method':<25} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Acc':<8} {'FPR@95':<8}")
print("-" * 80)
for k_label in ["1", "2", "3", "5", "full"]:
    for method in ["treatment", "baseline", "tfidf"]:
        m = results["early_detection"][k_label][method]
        method_name = {"treatment": "DeBERTa-FT + GRU", "baseline": "DeBERTa-vanilla + GRU", "tfidf": "TF-IDF + LR"}[method]
        print(f"{k_label:<6} {method_name:<25} {m['f1']:<8.4f} {m['precision']:<8.4f} {m['recall']:<8.4f} {m['accuracy']:<8.4f} {m['fpr_at_95tpr']:<8.4f}")
    print()

# Check pass criteria
for k_label in ["2", "3"]:
    t_f1 = results["early_detection"][k_label]["treatment"]["f1"]
    b_f1 = results["early_detection"][k_label]["baseline"]["f1"]
    delta = t_f1 - b_f1
    passed = "PASS" if delta > 0.05 else "FAIL"
    print(f"Pass criteria K={k_label}: Treatment F1={t_f1:.4f} - Baseline F1={b_f1:.4f} = {delta:.4f} [{passed}]")

# Save
Path('results').mkdir(exist_ok=True)
with open('results/plan_002_eval.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to results/plan_002_eval.json")

# Checkpoint paths
print("\nCheckpoint paths:")
print(f"  DeBERTa: checkpoints/plan_002/deberta_multitask/best/")
print(f"  GRU Treatment: checkpoints/plan_002/gru/treatment/best.pt")
print(f"  GRU Baseline: checkpoints/plan_002/gru/baseline/best.pt")

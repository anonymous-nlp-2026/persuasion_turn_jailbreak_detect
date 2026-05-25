#!/bin/bash
cd .
export CUDA_VISIBLE_DEVICES=2
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/.cache/huggingface
PYTHON=python

echo "=== Step 2: DeBERTa Multi-task (scrambled labels) ==="
$PYTHON src/train_deberta.py \
  --train_data data/final_v2/train_scrambled.jsonl \
  --val_data data/final_v2/val.jsonl \
  --model_name microsoft/deberta-v3-base \
  --output_dir checkpoints/plan003_deberta_scrambled \
  --max_length 256 \
  --batch_size 16 \
  --lr 2e-5 \
  --epochs 5 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --alpha 0.3 \
  --seed 42

if [ $? -ne 0 ]; then
    echo "FAILED: DeBERTa training"
    exit 1
fi

echo "=== Step 3: GRU Classifier (scrambled treatment) ==="
$PYTHON src/train_classifier.py \
  --train_data data/final_v2/train_scrambled.jsonl \
  --val_data data/final_v2/val.jsonl \
  --mode treatment \
  --deberta_checkpoint checkpoints/plan003_deberta_scrambled/best \
  --model_name microsoft/deberta-v3-base \
  --output_dir checkpoints/plan003_gru_scrambled \
  --max_length 256 \
  --batch_size 32 \
  --lr 1e-3 \
  --hidden_dim 256 \
  --num_layers 2 \
  --dropout 0.3 \
  --epochs 20 \
  --seed 42

if [ $? -ne 0 ]; then
    echo "FAILED: GRU training"
    exit 1
fi

echo "=== Step 4: Evaluation ==="
$PYTHON -c "
import sys, json, torch
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import AutoTokenizer

sys.path.insert(0, '.')
from src.models.deberta_multitask import DeBERTaMultiTask
from src.models.gru_classifier import GRUClassifier

device = torch.device('cuda:0')

# Load scrambled DeBERTa
model = DeBERTaMultiTask(model_name='microsoft/deberta-v3-base')
state = torch.load('checkpoints/plan003_deberta_scrambled/best/model.pt', map_location='cpu')
model.load_state_dict(state)
encoder = model.deberta.to(device).eval()
for p in encoder.parameters():
    p.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
embed_dim = encoder.config.hidden_size

# Load GRU
gru = GRUClassifier(input_dim=embed_dim, hidden_dim=256, num_layers=2, dropout=0.3)
gru.load_state_dict(torch.load('checkpoints/plan003_gru_scrambled/treatment/best.pt', map_location='cpu'))
gru = gru.to(device).eval()

# Load test data
convs = []
with open('data/final_v2/test.jsonl') as f:
    for line in f:
        c = json.loads(line.strip())
        user_turns = [t['content'] for t in c['turns'] if t['role'] == 'user']
        label = 1 if c['label'] == 'jailbreak' else 0
        convs.append({'turns': user_turns, 'label': label, 'id': c['conversation_id']})

print(f'Test conversations: {len(convs)}')

results = {}
K_values = [1, 2, 3, 5, None]

for k in K_values:
    k_label = str(k) if k is not None else 'full'
    all_embs = []
    all_labels = []
    all_lengths = []

    for conv in convs:
        turns = conv['turns'][:k] if k is not None else conv['turns']
        if len(turns) == 0:
            turns = ['']
        enc = tokenizer(turns, max_length=256, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = encoder(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
            embs = outputs.last_hidden_state[:, 0, :].cpu()
        all_embs.append(embs)
        all_labels.append(conv['label'])
        all_lengths.append(len(turns))

    # GRU evaluation
    max_len = max(all_lengths)
    padded = torch.zeros(len(all_embs), max_len, embed_dim)
    for i, e in enumerate(all_embs):
        padded[i, :e.size(0), :] = e
    lengths_t = torch.tensor(all_lengths, dtype=torch.long)
    labels_np = np.array(all_labels)

    with torch.no_grad():
        logits = gru(padded.to(device), lengths_t.to(device))
        preds = logits.argmax(-1).cpu().numpy()
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    f1 = f1_score(labels_np, preds, zero_division=0)
    prec = precision_score(labels_np, preds, zero_division=0)
    rec = recall_score(labels_np, preds, zero_division=0)
    acc = accuracy_score(labels_np, preds)

    # FPR@95TPR
    benign_probs = probs[labels_np == 0]
    jb_probs = probs[labels_np == 1]
    if len(jb_probs) > 0 and len(benign_probs) > 0:
        thresholds = np.sort(jb_probs)
        idx = max(0, int(np.floor(len(thresholds) * 0.05)) - 1)
        thresh = thresholds[idx]
        fpr = (benign_probs >= thresh).mean()
    else:
        fpr = 0.0

    results[k_label] = {'f1': f1, 'precision': prec, 'recall': rec, 'accuracy': acc, 'fpr_at_95tpr': float(fpr)}
    print(f'K={k_label}: F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f} Acc={acc:.4f} FPR@95={fpr:.4f}')

# Save
Path('results').mkdir(exist_ok=True)
with open('results/plan_003_scrambled_eval.json', 'w') as f:
    json.dump({'scrambled_ablation': results, 'note': 'Scrambled persuasion_strategy labels on train jailbreak turns, seed=42'}, f, indent=2)
print('\nResults saved to results/plan_003_scrambled_eval.json')
"

echo "=== plan_003 COMPLETE ==="

#!/bin/bash
cd .
export CUDA_VISIBLE_DEVICES=2
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/.cache/huggingface
PYTHON=python

echo "=== Step 4: Evaluation ==="
$PYTHON -u -c "
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
print('Loading scrambled DeBERTa...')
model = DeBERTaMultiTask(model_name='microsoft/deberta-v3-base')
state = torch.load('checkpoints/plan003_deberta_scrambled/best/model.pt', map_location='cpu')
model.load_state_dict(state)
encoder = model.deberta.to(device).eval()
for p in encoder.parameters():
    p.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
embed_dim = encoder.config.hidden_size

# Load GRU
print('Loading GRU...')
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
        fpr = float((benign_probs >= thresh).mean())
    else:
        fpr = 0.0

    results[k_label] = {'f1': float(f1), 'precision': float(prec), 'recall': float(rec), 'accuracy': float(acc), 'fpr_at_95tpr': fpr}
    print(f'K={k_label}: F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f} Acc={acc:.4f} FPR@95={fpr:.4f}')

# Save
Path('results').mkdir(exist_ok=True)
output = {
    'experiment': 'plan_003_scrambled_label_ablation',
    'description': 'DeBERTa fine-tuned with scrambled persuasion_strategy labels on jailbreak turns (seed=42). Tests whether strategy labels carry real signal.',
    'deberta_training': {
        'epochs_completed': 4,
        'best_epoch': 3,
        'best_val_loss': 1.0147,
        'val_persuasion_acc': 0.6437,
        'val_intent_acc': 0.9674,
    },
    'early_detection': results,
}
with open('results/plan_003_scrambled_eval.json', 'w') as f:
    json.dump(output, f, indent=2)
print('\nResults saved to results/plan_003_scrambled_eval.json')

# Summary table
print('\n' + '='*70)
print(f\"{'plan_003: Scrambled-Label Ablation Results':^70}\")
print('='*70)
print(f\"{'K':<6} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Acc':<8} {'FPR@95':<8}\")
print('-'*46)
for k_label in ['1', '2', '3', '5', 'full']:
    m = results[k_label]
    print(f\"{k_label:<6} {m['f1']:<8.4f} {m['precision']:<8.4f} {m['recall']:<8.4f} {m['accuracy']:<8.4f} {m['fpr_at_95tpr']:<8.4f}\")
"

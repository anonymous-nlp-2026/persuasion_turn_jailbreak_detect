import torch
import torch.nn as nn
import sys
sys.path.insert(0, ".")
from src.models.gru_classifier import GRUClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = GRUClassifier(input_dim=768, hidden_dim=256, num_layers=2, dropout=0.3, num_classes=2)
model.load_state_dict(torch.load("checkpoints/baseline_gru/best.pt", map_location=device, weights_only=False))
model.to(device)
model.eval()

test_data = torch.load("data/embeddings/baseline_test.pt", map_location="cpu", weights_only=False)

def evaluate_subset(name, subset):
    if len(subset) == 0:
        print(f"| {name} | 0 | - | - | - | - | - |")
        return
    
    all_preds = []
    all_labels = []
    
    for sample in subset:
        emb = sample["embeddings"].unsqueeze(0).to(device)  # (1, N, 768)
        length = torch.tensor([emb.size(1)], dtype=torch.long).to(device)
        label = sample["label"]
        
        with torch.no_grad():
            logits = model(emb, length)
            pred = logits.argmax(-1).item()
        
        all_preds.append(pred)
        all_labels.append(label)
    
    y_true = all_labels
    y_pred = all_preds
    
    acc = accuracy_score(y_true, y_pred)
    
    # For subsets with only one class, handle metrics carefully
    unique_labels = set(y_true)
    if len(unique_labels) == 1:
        if unique_labels == {0}:
            # All benign: FPR = predicted positive / total
            fp = sum(1 for p in y_pred if p == 1)
            fpr = fp / len(y_pred)
            print(f"| {name} | {len(subset)} | N/A (single class) | {acc:.4f} | N/A | N/A | {fpr:.4f} |")
        else:
            # All jailbreak: recall = predicted positive / total
            rec = recall_score(y_true, y_pred, pos_label=1)
            print(f"| {name} | {len(subset)} | N/A (single class) | {acc:.4f} | N/A | {rec:.4f} | N/A |")
        return
    
    f1 = f1_score(y_true, y_pred, pos_label=1)
    prec = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    print(f"| {name} | {len(subset)} | {f1:.4f} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {fpr:.4f} |")
    
    # Print confusion matrix detail
    print(f"  -> CM: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

# Split by attack type
persuasion = [d for d in test_data if d["attack_type"] == "persuasion-based"]
cipher = [d for d in test_data if d["attack_type"] == "cipher-based"]
benign = [d for d in test_data if d["attack_type"] == "benign"]

print("| Subset | N | F1 | Accuracy | Precision | Recall | FPR |")
print("|--------|---|----|---------|-----------|---------|----|")

evaluate_subset("Full test", test_data)
evaluate_subset("Persuasion-based (jailbreak)", persuasion)
evaluate_subset("Cipher-based (jailbreak)", cipher)
evaluate_subset("Benign only", benign)
evaluate_subset("Persuasion + Benign", persuasion + benign)

# Summary
print("\n--- Key Finding ---")
p_preds = []
c_preds = []
for s in persuasion:
    emb = s["embeddings"].unsqueeze(0).to(device)
    length = torch.tensor([emb.size(1)], dtype=torch.long).to(device)
    with torch.no_grad():
        pred = model(emb, length).argmax(-1).item()
    p_preds.append(pred)

for s in cipher:
    emb = s["embeddings"].unsqueeze(0).to(device)
    length = torch.tensor([emb.size(1)], dtype=torch.long).to(device)
    with torch.no_grad():
        pred = model(emb, length).argmax(-1).item()
    c_preds.append(pred)

p_recall = sum(p_preds) / len(p_preds)
c_recall = sum(c_preds) / len(c_preds)
print(f"Persuasion recall: {p_recall:.4f} ({sum(p_preds)}/{len(p_preds)})")
print(f"Cipher recall:     {c_recall:.4f} ({sum(c_preds)}/{len(c_preds)})")
print(f"Gap:               {c_recall - p_recall:.4f}")

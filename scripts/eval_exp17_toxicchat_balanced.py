import sys, json, numpy as np, torch
from pathlib import Path
from sklearn.metrics import f1_score

sys.path.insert(0, ".")
from src.models.gru_classifier import GRUClassifier
from src.models.deberta_multitask import DeBERTaMultiTask
from transformers import AutoTokenizer, AutoModel

PROJ = Path(".")
DATA = PROJ / "data/mhj/toxicchat_eval.jsonl"
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
DEVICE = torch.device("cpu")
SEEDS = [42, 123, 456]

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def extract_user_turns(conv):
    return [t["content"] for t in conv["turns"] if t["role"] == "user"]

def get_label(conv):
    return 1 if conv["label"] == "jailbreak" else 0

def load_encoder(deberta_ckpt, device):
    model = DeBERTaMultiTask(model_name=MODEL_NAME)
    state_dict = torch.load(Path(deberta_ckpt) / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    encoder = model.deberta
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder.float().to(device)

def evaluate(encoder, gru, tokenizer, convs, device):
    gru.eval()
    preds, labels = [], []
    for c in convs:
        turns = extract_user_turns(c)
        if not turns:
            turns = [""]
        enc = tokenizer(turns, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            embs = out.last_hidden_state[:, 0, :].unsqueeze(0)
            lengths = torch.tensor([len(turns)], dtype=torch.long).to(device)
            logits = gru(embs, lengths)
            pred = logits.argmax(dim=1).item()
        preds.append(pred)
        labels.append(get_label(c))
    return float(f1_score(np.array(labels), np.array(preds), average="macro"))

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    convs = load_jsonl(DATA)
    n_jb = sum(1 for c in convs if c["label"] == "jailbreak")
    print(f"Data: {DATA.name} ({len(convs)} samples, {n_jb}jb/{len(convs)-n_jb}bn)")

    f1s = []
    for seed in SEEDS:
        deberta_ckpt = PROJ / f"checkpoints/exp17/9class_seed{seed}/deberta_multitask/best"
        gru_ckpt = PROJ / f"checkpoints/exp17/9class_seed{seed}/treatment/best.pt"
        encoder = load_encoder(deberta_ckpt, DEVICE)
        gru = GRUClassifier(input_dim=encoder.config.hidden_size, hidden_dim=256, num_layers=2, dropout=0.3)
        gru.load_state_dict(torch.load(gru_ckpt, map_location="cpu"))
        gru.to(DEVICE)
        gru.eval()
        f1 = evaluate(encoder, gru, tokenizer, convs, DEVICE)
        f1s.append(f1)
        print(f"  seed{seed}: {f1:.4f}")
        del encoder, gru

    print(f"  mean±std: {np.mean(f1s):.4f}±{np.std(f1s):.4f}")

if __name__ == "__main__":
    main()

import json
import random
import re
from collections import Counter
from datasets import load_dataset

# === Config ===
OUTPUT_PATH = "data/generated/wildchat_benign.jsonl"
SEED = 42
TARGET_N = 250
MIN_USER_TURNS = 4
MIN_AVG_WORDS = 20
MAX_CANDIDATES = 2000  # collect up to this many, then sample

# === Jailbreak / prompt injection keywords ===
JAILBREAK_KEYWORDS = [
    "ignore previous instructions", "ignore all previous", "disregard previous",
    "forget your instructions", "you are now", "act as if", "pretend you are",
    "jailbreak", "DAN", "do anything now", "bypass", "override your",
    "ignore your programming", "ignore your training", "system prompt",
    "you have no restrictions", "no ethical guidelines", "hypothetically",
    "for educational purposes only", "in a fictional scenario",
    "roleplay as", "evil mode", "developer mode", "god mode",
    "ignore safety", "ignore content policy", "opposite day",
]
JAILBREAK_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in JAILBREAK_KEYWORDS), re.IGNORECASE
)

def extract_topic(first_user_msg, max_words=5):
    """Simple topic extraction from first user message."""
    text = first_user_msg.strip()
    # Remove common greetings
    text = re.sub(r'^(hi|hello|hey|good morning|good afternoon|good evening)[,!.\s]*', '', text, flags=re.IGNORECASE)
    # Take first sentence or first N words
    first_sent = re.split(r'[.!?\n]', text)[0].strip()
    words = first_sent.split()[:max_words]
    topic = " ".join(words).lower().rstrip(",;:")
    if len(topic) < 3:
        topic = " ".join(text.split()[:max_words]).lower()
    return topic[:80]

def is_valid_conversation(example):
    """Check if conversation meets our criteria."""
    # Multi-turn: >= 4 user turns
    if example.get("turn", 0) < MIN_USER_TURNS:
        return False
    # English
    if example.get("language", "") != "English":
        return False
    # Not toxic
    if example.get("toxic", True):
        return False
    
    conv = example.get("conversation", [])
    if not conv:
        return False
    
    # Extract user messages
    user_msgs = [m["content"] for m in conv if m.get("role") == "user"]
    if len(user_msgs) < MIN_USER_TURNS:
        return False
    
    # Average word count >= 20
    avg_words = sum(len(m.split()) for m in user_msgs) / len(user_msgs)
    if avg_words < MIN_AVG_WORDS:
        return False
    
    # No jailbreak patterns in any message
    full_text = " ".join(m["content"] for m in conv)
    if JAILBREAK_PATTERN.search(full_text):
        return False
    
    # ASCII ratio > 0.9 (extra English check)
    ascii_count = sum(1 for c in full_text if ord(c) < 128)
    if len(full_text) > 0 and ascii_count / len(full_text) < 0.9:
        return False
    
    return True

def convert_to_project_format(example, idx):
    """Convert WildChat example to project format."""
    conv = example["conversation"]
    turns = []
    for msg in conv:
        turn = {"role": msg["role"], "content": msg["content"]}
        if msg["role"] == "user":
            turn["intended_strategy"] = 0
        turns.append(turn)
    
    first_user = next((m["content"] for m in conv if m["role"] == "user"), "")
    topic = extract_topic(first_user)
    
    return {
        "conversation_id": f"wildchat_{idx:04d}",
        "source": "wildchat",
        "turns": turns,
        "label": "benign",
        "topic": topic,
        "num_turns": len([t for t in turns if t["role"] == "user"]),
    }

# === Main ===
print("Loading WildChat-1M in streaming mode...")
ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

candidates = []
total_seen = 0
filtered_reasons = Counter()

print("Filtering...")
for example in ds:
    total_seen += 1
    
    if total_seen % 10000 == 0:
        print(f"  Seen {total_seen}, candidates: {len(candidates)}")
    
    # Quick filters first
    if example.get("turn", 0) < MIN_USER_TURNS:
        filtered_reasons["< 4 user turns"] += 1
        continue
    if example.get("language", "") != "English":
        filtered_reasons["not English"] += 1
        continue
    if example.get("toxic", True):
        filtered_reasons["toxic"] += 1
        continue
    
    if is_valid_conversation(example):
        candidates.append(example)
        if len(candidates) >= MAX_CANDIDATES:
            print(f"  Reached {MAX_CANDIDATES} candidates at {total_seen} seen. Stopping.")
            break
    else:
        # Detailed reason already counted above, this catches remaining
        filtered_reasons["other (length/jailbreak/ascii)"] += 1

print(f"\nTotal seen: {total_seen}")
print(f"Candidates after filtering: {len(candidates)}")
print(f"Filter reasons: {dict(filtered_reasons)}")

# === Sample 250 ===
random.seed(SEED)
if len(candidates) < TARGET_N:
    print(f"WARNING: Only {len(candidates)} candidates, using all")
    sampled = candidates
else:
    sampled = random.sample(candidates, TARGET_N)

# === Convert and save ===
results = []
for i, ex in enumerate(sampled):
    results.append(convert_to_project_format(ex, i))

with open(OUTPUT_PATH, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nSaved {len(results)} conversations to {OUTPUT_PATH}")

# === Statistics ===
turn_counts = [r["num_turns"] for r in results]
user_word_counts = []
for r in results:
    user_msgs = [t["content"] for t in r["turns"] if t["role"] == "user"]
    avg_wc = sum(len(m.split()) for m in user_msgs) / len(user_msgs)
    user_word_counts.append(avg_wc)

topics = Counter(r["topic"] for r in results)

print(f"\n=== Statistics ===")
print(f"Samples: {len(results)}")
print(f"Avg user turns: {sum(turn_counts)/len(turn_counts):.1f}")
print(f"Min/Max user turns: {min(turn_counts)}/{max(turn_counts)}")
print(f"Avg user turn word count: {sum(user_word_counts)/len(user_word_counts):.1f}")
print(f"\nTop 20 topics:")
for topic, count in topics.most_common(20):
    print(f"  {count:3d}  {topic}")

# === TF-IDF Sanity Check ===
print("\n=== TF-IDF Sanity Check ===")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
import numpy as np

# Load crescendo jailbreak data
crescendo = []
with open("data/generated/crescendo_all.jsonl") as f:
    for line in f:
        crescendo.append(json.loads(line))

print(f"Crescendo jailbreak: {len(crescendo)} conversations")
print(f"WildChat benign: {len(results)} conversations")

# Extract all user text per conversation
def get_user_text(conv_turns):
    return " ".join(t["content"] for t in conv_turns if t["role"] == "user")

texts = []
labels = []

for r in results:
    texts.append(get_user_text(r["turns"]))
    labels.append(0)  # benign

for c in crescendo:
    texts.append(get_user_text(c["turns"]))
    labels.append(1)  # jailbreak

labels = np.array(labels)

# TF-IDF + Logistic Regression with 5-fold CV
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
X = tfidf.fit_transform(texts)

clf = LogisticRegression(max_iter=1000, random_state=42)
y_pred = cross_val_predict(clf, X, labels, cv=5)

print("\nClassification Report (5-fold CV):")
print(classification_report(labels, y_pred, target_names=["benign", "jailbreak"]))

from sklearn.metrics import f1_score
f1 = f1_score(labels, y_pred, average="macro")
print(f"Macro F1: {f1:.4f}")

# Top features
clf.fit(X, labels)
feature_names = tfidf.get_feature_names_out()
coefs = clf.coef_[0]

# Top 10 for each class
top_jailbreak_idx = np.argsort(coefs)[-10:][::-1]
top_benign_idx = np.argsort(coefs)[:10]

print("\nTop 10 features → Jailbreak (Class B):")
for idx in top_jailbreak_idx:
    print(f"  {feature_names[idx]:30s}  coef={coefs[idx]:.4f}")

print("\nTop 10 features → Benign (Class A):")
for idx in top_benign_idx:
    print(f"  {feature_names[idx]:30s}  coef={coefs[idx]:.4f}")

if f1 < 0.80:
    print(f"\n*** F1 = {f1:.4f} < 0.80 — MAJOR BREAKTHROUGH! TF-IDF confound significantly reduced! ***")
elif f1 < 0.90:
    print(f"\n*** F1 = {f1:.4f} < 0.90 — Significant improvement over LLM-generated benign data ***")
else:
    print(f"\n*** F1 = {f1:.4f} >= 0.90 — Still separable, but check if improvement over previous ***")


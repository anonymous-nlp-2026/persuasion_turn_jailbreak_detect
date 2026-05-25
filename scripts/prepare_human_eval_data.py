"""Prepare human jailbreak evaluation data from ToxicChat."""
import json
import pandas as pd
import random
import sys

random.seed(42)

# Load both splits
train = pd.read_csv('./data/mhj/toxicchat_train.csv')
test = pd.read_csv('./data/mhj/toxicchat_test.csv')
df = pd.concat([train, test], ignore_index=True)

print(f"Total ToxicChat examples: {len(df)}")

# Get jailbreak examples
jb_df = df[df['jailbreaking'] == 1].copy()
print(f"Jailbreak examples: {len(jb_df)}")

# Get benign examples (not toxic, not jailbreaking)
benign_df = df[(df['jailbreaking'] == 0) & (df['toxicity'] == 0)].copy()
print(f"Available benign examples: {len(benign_df)}")

# Sample benign to match jailbreak count (balanced dataset)
n_benign = min(len(jb_df), len(benign_df))
benign_sample = benign_df.sample(n=n_benign, random_state=42)
print(f"Sampled benign: {n_benign}")

# Convert to our conversation format
conversations = []

for _, row in jb_df.iterrows():
    turns = [{"role": "user", "content": str(row['user_input'])}]
    if pd.notna(row['model_output']) and str(row['model_output']).strip():
        turns.append({"role": "assistant", "content": str(row['model_output'])})
    conv = {
        "conversation_id": f"toxicchat_jb_{row['conv_id']}",
        "turns": turns,
        "label": "jailbreak",
        "source": "toxicchat-human",
        "attack_type": "human_jailbreak"
    }
    conversations.append(conv)

for _, row in benign_sample.iterrows():
    turns = [{"role": "user", "content": str(row['user_input'])}]
    if pd.notna(row['model_output']) and str(row['model_output']).strip():
        turns.append({"role": "assistant", "content": str(row['model_output'])})
    conv = {
        "conversation_id": f"toxicchat_bn_{row['conv_id']}",
        "turns": turns,
        "label": "benign",
        "source": "toxicchat-human",
        "attack_type": "benign"
    }
    conversations.append(conv)

random.shuffle(conversations)
print(f"\nTotal conversations: {len(conversations)}")
print(f"  Jailbreak: {sum(1 for c in conversations if c['label']=='jailbreak')}")
print(f"  Benign: {sum(1 for c in conversations if c['label']=='benign')}")

# Also create a version using plan_002 benign as control
plan002_benign = []
with open('./data/plan_002_splits/test.jsonl') as f:
    for line in f:
        conv = json.loads(line.strip())
        if conv['label'] == 'benign':
            plan002_benign.append(conv)

print(f"\nPlan_002 benign conversations: {len(plan002_benign)}")

# Version 1: ToxicChat jailbreak + ToxicChat benign (matched single-turn)
outpath1 = './data/mhj/toxicchat_eval.jsonl'
with open(outpath1, 'w') as f:
    for conv in conversations:
        f.write(json.dumps(conv) + '\n')
print(f"\nSaved balanced dataset to {outpath1}")

# Version 2: ToxicChat jailbreak + plan_002 benign (mixed turn lengths)
convs_v2 = [c for c in conversations if c['label'] == 'jailbreak'] + plan002_benign
random.shuffle(convs_v2)
outpath2 = './data/mhj/toxicchat_plan002_eval.jsonl'
with open(outpath2, 'w') as f:
    for conv in convs_v2:
        f.write(json.dumps(conv) + '\n')
print(f"Saved mixed dataset to {outpath2} ({len(convs_v2)} conversations)")

# Stats
jb_lens = [len(row['user_input'].split()) for _, row in jb_df.iterrows()]
print(f"\nJailbreak prompt length stats: mean={sum(jb_lens)/len(jb_lens):.0f} words, "
      f"min={min(jb_lens)}, max={max(jb_lens)}")


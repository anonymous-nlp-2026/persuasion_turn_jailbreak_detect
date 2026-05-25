import json, random
from sklearn.model_selection import train_test_split
from pathlib import Path

# Load jailbreak data
with open('data/generated/crescendo_all.jsonl') as f:
    jailbreak_raw = [json.loads(l) for l in f]

# Load benign data
benign_raw = []
for fpath in ['data/generated/benign_v4c_full_gpu0.jsonl', 'data/generated/benign_v4c_full_gpu2.jsonl']:
    with open(fpath) as f:
        benign_raw.extend([json.loads(l) for l in f])

print(f'Jailbreak: {len(jailbreak_raw)}, Benign: {len(benign_raw)}')

# Normalize jailbreak data
all_data = []
for item in jailbreak_raw:
    turns = []
    for t in item['turns']:
        turn = {'role': t['role'], 'content': t['content']}
        if t['role'] == 'user':
            turn['persuasion_strategy'] = t['intended_strategy']
        turns.append(turn)
    all_data.append({
        'conversation_id': item['conv_id'],
        'turns': turns,
        'label': 'jailbreak',
        'source': item.get('playbook_name', 'crescendo'),
    })

# Normalize benign data
for item in benign_raw:
    turns = []
    for t in item['turns']:
        turn = {'role': t['role'], 'content': t['content']}
        if t['role'] == 'user':
            turn['persuasion_strategy'] = 7  # Direct Request
        turns.append(turn)
    all_data.append({
        'conversation_id': item['conversation_id'],
        'turns': turns,
        'label': 'benign',
        'source': item.get('source', 'benign'),
    })

print(f'Total: {len(all_data)}')

# Stratified split: 70/15/15
labels = [d['label'] for d in all_data]
train_val, test = train_test_split(all_data, test_size=0.15, random_state=42, stratify=labels)
train_val_labels = [d['label'] for d in train_val]
train, val = train_test_split(train_val, test_size=0.176, random_state=42, stratify=train_val_labels)

print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')
print(f'Train: jb={sum(1 for d in train if d["label"]=="jailbreak")}, bn={sum(1 for d in train if d["label"]=="benign")}')
print(f'Val: jb={sum(1 for d in val if d["label"]=="jailbreak")}, bn={sum(1 for d in val if d["label"]=="benign")}')
print(f'Test: jb={sum(1 for d in test if d["label"]=="jailbreak")}, bn={sum(1 for d in test if d["label"]=="benign")}')

# Count user turns per split
for split_name, split_data in [('Train', train), ('Val', val), ('Test', test)]:
    user_turns = sum(1 for d in split_data for t in d['turns'] if t['role'] == 'user')
    print(f'{split_name} user turns: {user_turns}')

# Save
out_dir = Path('data/final')
out_dir.mkdir(parents=True, exist_ok=True)

for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
    with open(out_dir / f'{split_name}.jsonl', 'w') as f:
        for item in split_data:
            f.write(json.dumps(item) + '\n')

print('Done. Saved to data/final/')

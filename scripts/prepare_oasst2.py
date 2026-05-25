import json
import random
from collections import defaultdict
from datasets import load_dataset

print("Loading oasst2...")
ds = load_dataset("OpenAssistant/oasst2", split="train")
print(f"Total messages: {len(ds)}")

# Index messages by id and group by tree
msg_by_id = {}
children_by_parent = defaultdict(list)
roots = []

for row in ds:
    msg = dict(row)
    msg_by_id[msg["message_id"]] = msg
    if msg["parent_id"] is None:
        roots.append(msg)
    else:
        children_by_parent[msg["parent_id"]].append(msg)

print(f"Total trees: {len(roots)}")

# Build conversations: from each root, follow highest-rank child at each step
def build_conversation(root):
    conv = [root]
    current = root
    while True:
        kids = children_by_parent.get(current["message_id"], [])
        if not kids:
            break
        # Sort by rank (lower rank = better), pick best
        ranked = [k for k in kids if k.get("rank") is not None]
        if ranked:
            ranked.sort(key=lambda x: x["rank"])
            best = ranked[0]
        else:
            # No rank info, pick first child
            best = kids[0]
        conv.append(best)
        current = best
    return conv

conversations = []
for root in roots:
    conv = build_conversation(root)
    conversations.append(conv)

print(f"Total conversations built: {len(conversations)}")

# Filter: English, multi-turn (>=3 user messages = >=6 messages alternating)
JAILBREAK_KEYWORDS = [
    "jailbreak", "ignore previous", "ignore all", "DAN", "pretend you are",
    "bypass", "hack into", "exploit", "generate harmful", "write malware",
    "make a bomb", "how to kill"
]

def is_benign(conv):
    text = " ".join(m["text"].lower() for m in conv)
    return not any(kw.lower() in text for kw in JAILBREAK_KEYWORDS)

def is_english(conv):
    return all(m.get("lang") == "en" for m in conv)

def count_user_turns(conv):
    return sum(1 for m in conv if m["role"] == "prompter")

filtered = []
for conv in conversations:
    if not is_english(conv):
        continue
    if count_user_turns(conv) < 3:
        continue
    if not is_benign(conv):
        continue
    filtered.append(conv)

print(f"After filtering (English, >=3 user turns, benign): {len(filtered)}")

# Convert to project format
def conv_to_record(conv, idx):
    user_turns = []
    for m in conv:
        if m["role"] == "prompter":
            user_turns.append({"role": "user", "content": m["text"]})
    return {
        "conversation_id": f"oasst2_{conv[0]['message_tree_id'][:12]}",
        "source": "oasst2",
        "label": "benign",
        "turns": user_turns,
        "num_turns": len(user_turns),
        "metadata": {
            "message_tree_id": conv[0]["message_tree_id"],
            "total_messages": len(conv),
        },
        "attack_type": "benign"
    }

records = [conv_to_record(c, i) for i, c in enumerate(filtered)]

# Shuffle and split
random.seed(42)
random.shuffle(records)

eval_set = records[:200]
adapt_set = records[200:250]

# Save
out_dir = "./data"
eval_path = f"{out_dir}/oasst2_benign_eval_200.json"
adapt_path = f"{out_dir}/oasst2_benign_adapt_50.json"

with open(eval_path, "w") as f:
    for r in eval_set:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(adapt_path, "w") as f:
    for r in adapt_set:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nSaved {len(eval_set)} eval records to {eval_path}")
print(f"Saved {len(adapt_set)} adapt records to {adapt_path}")

# Quality report
turn_counts = [r["num_turns"] for r in records]
avg_turns = sum(turn_counts) / len(turn_counts)
print(f"\n=== Quality Report ===")
print(f"Dataset: OpenAssistant/oasst2 (real crowdsourced human conversations)")
print(f"Total filtered pool: {len(records)}")
print(f"Average user turns: {avg_turns:.1f}")
print(f"\nFirst 3 samples (first user message):")
for i, r in enumerate(eval_set[:3]):
    msg = r["turns"][0]["content"][:150]
    print(f"  [{i+1}] ({r['num_turns']} turns) {msg}")

print(f"\nEval set: {len(eval_set)} conversations")
print(f"Adapt set: {len(adapt_set)} conversations")

# Turn distribution
from collections import Counter
dist = Counter(r["num_turns"] for r in records)
print(f"\nTurn distribution: {dict(sorted(dist.items()))}")

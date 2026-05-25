"""
exp51: Prepare OOD benign multi-turn conversation data.
Uses HuggingFaceH4/ultrachat_200k via hf-mirror.com.
"""
import json, random, re, hashlib, sys, os
sys.stdout.reconfigure(line_buffering=True)
random.seed(42)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

JAILBREAK_PATTERNS = re.compile(
    r"ignore previous|ignore all|\bDAN\b|jailbreak|pretend you are|act as if|"
    r"you are now|bypass|do anything now|ignore the rules|override|"
    r"disregard|forget your|new persona|evil mode|developer mode|"
    r"no restrictions|without any filter|unfiltered",
    re.IGNORECASE
)
OUT_DIR = "./data"

print("Loading HuggingFaceH4/ultrachat_200k via hf-mirror...", flush=True)
from datasets import load_dataset
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)

candidates = []
seen = 0

for item in ds:
    seen += 1
    if seen % 5000 == 0:
        print(f"  Scanned {seen}, found {len(candidates)} candidates...", flush=True)
    
    messages = item.get("messages", [])
    if not messages:
        continue
    
    user_turns = [m for m in messages if m.get("role") == "user"]
    if len(user_turns) < 3:
        continue
    
    first_msg = user_turns[0].get("content", "")
    if not first_msg or len(first_msg) < 10:
        continue
    
    ascii_ratio = sum(1 for c in first_msg if ord(c) < 128) / max(len(first_msg), 1)
    if ascii_ratio < 0.85:
        continue
    
    all_text = " ".join(t.get("content", "") for t in user_turns)
    if JAILBREAK_PATTERNS.search(all_text):
        continue
    if len(all_text) > 20000:
        continue
    
    conv_id = "ultrachat_" + hashlib.md5(all_text[:200].encode()).hexdigest()[:12]
    entry = {
        "conversation_id": conv_id,
        "source": "ultrachat",
        "label": "benign",
        "turns": [{"role": "user", "content": t["content"]} for t in user_turns],
        "num_turns": len(user_turns),
        "metadata": {"prompt_id": str(item.get("prompt_id", "unknown"))},
        "attack_type": "benign"
    }
    candidates.append(entry)
    if len(candidates) >= 500:
        break

print(f"Scanned {seen} total, {len(candidates)} candidates after filtering.", flush=True)

if len(candidates) < 200:
    print(f"WARNING: Only {len(candidates)} candidates, using all.")
    selected = candidates
else:
    selected = random.sample(candidates, 200)

selected.sort(key=lambda x: x["conversation_id"])

out_200 = f"{OUT_DIR}/lmsys_benign_200.json"
with open(out_200, "w") as f:
    for item in selected:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved {len(selected)} to {out_200}", flush=True)

adapt = selected[:10]
out_10 = f"{OUT_DIR}/lmsys_benign_10_adapt.json"
with open(out_10, "w") as f:
    for item in adapt:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved {len(adapt)} to {out_10}", flush=True)

tc = [i["num_turns"] for i in selected]
print(f"\n=== Statistics ===")
print(f"Total: {len(selected)}")
print(f"Avg turns: {sum(tc)/len(tc):.1f}")
print(f"Min/Max: {min(tc)}/{max(tc)}")
print(f"Median: {sorted(tc)[len(tc)//2]}")
from collections import Counter
dist = Counter(tc)
print(f"Turn distribution: {dict(sorted(dist.items()))}")

print(f"\n=== Samples ===")
for i, item in enumerate(random.sample(selected, min(5, len(selected)))):
    print(f"\n--- {item['conversation_id']} ({item['num_turns']} turns) ---")
    for t in item["turns"][:3]:
        print(f"  [{t['role']}] {t['content'][:120].replace(chr(10),' ')}...")

print("\nDone!", flush=True)

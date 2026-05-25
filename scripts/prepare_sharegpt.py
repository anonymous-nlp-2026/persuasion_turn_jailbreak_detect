#!/usr/bin/env python3
"""Prepare ShareGPT benign conversation data for exp66.

Parses downloaded ShareGPT data (partial or full), filters to
English multi-turn benign conversations, saves eval (200) + adapt (50) sets.
"""

import json, random, re

SHAREGPT_FILE = "./data/sharegpt_v3.json"
OUT_DIR = "./data"

JAILBREAK_KEYWORDS = [
    "ignore previous instructions", "ignore all previous", "disregard previous",
    "forget your instructions", "you are now", "act as if", "pretend you are",
    "jailbreak", "DAN", "do anything now", "bypass", "override your",
    "ignore your programming", "ignore your training", "system prompt",
    "you have no restrictions", "no ethical guidelines",
    "roleplay as", "evil mode", "developer mode", "god mode",
    "ignore safety", "ignore content policy", "opposite day",
    "LEO", "no restrictions", "unlimited", "unfiltered",
]
JAILBREAK_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in JAILBREAK_KEYWORDS), re.IGNORECASE
)


def parse_partial_json(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    # Try full parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # Find last valid closing brace for a conversation object
    idx = len(content) - 1
    while idx > 0:
        idx = content.rfind('},', 0, idx)
        if idx == -1:
            break
        candidate = content[:idx+1] + '\n]'
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            idx -= 1
    raise RuntimeError("Could not parse JSON file")


def count_user_turns(conv):
    return sum(1 for msg in conv.get("conversations", []) if msg.get("from") == "human")


def is_benign(conv):
    text = " ".join(
        msg.get("value", "") for msg in conv.get("conversations", [])
        if msg.get("from") == "human"
    )
    return not JAILBREAK_PATTERN.search(text)


def looks_english(text):
    if not text:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / max(len(text), 1) > 0.8


def conv_to_record(conv):
    user_turns = []
    for msg in conv.get("conversations", []):
        if msg.get("from") == "human":
            content = msg.get("value", "").strip()
            if content:
                user_turns.append({"role": "user", "content": content})
    return {
        "conversation_id": f"sharegpt_{conv.get('id', 'unknown')}",
        "source": "sharegpt",
        "label": "benign",
        "turns": user_turns,
        "num_turns": len(user_turns),
        "attack_type": "benign",
    }


if __name__ == "__main__":
    print("Loading ShareGPT data...", flush=True)
    data = parse_partial_json(SHAREGPT_FILE)
    print(f"Total conversations: {len(data)}", flush=True)

    filtered = []
    for conv in data:
        if count_user_turns(conv) < 3:
            continue
        if not is_benign(conv):
            continue
        user_text = " ".join(
            msg.get("value", "")[:200] for msg in conv.get("conversations", [])
            if msg.get("from") == "human"
        )
        if not looks_english(user_text):
            continue
        filtered.append(conv)

    print(f"After filtering (>=3 user turns, benign, English): {len(filtered)}", flush=True)

    records = [conv_to_record(c) for c in filtered]

    random.seed(42)
    random.shuffle(records)

    eval_set = records[:200]
    adapt_set = records[200:250]

    eval_path = f"{OUT_DIR}/sharegpt_benign_eval_200.json"
    adapt_path = f"{OUT_DIR}/sharegpt_benign_adapt_50.json"

    for path, data_list in [(eval_path, eval_set), (adapt_path, adapt_set)]:
        with open(path, "w") as f:
            for r in data_list:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(eval_set)} eval to {eval_path}", flush=True)
    print(f"Saved {len(adapt_set)} adapt to {adapt_path}", flush=True)

    turn_counts = [r["num_turns"] for r in records[:250]]
    avg_turns = sum(turn_counts) / len(turn_counts)
    print(f"Pool size: {len(records)}, avg user turns: {avg_turns:.1f}", flush=True)
    for i, r in enumerate(eval_set[:3]):
        msg = r["turns"][0]["content"][:120]
        print(f"  [{i+1}] ({r['num_turns']} turns) {msg}", flush=True)

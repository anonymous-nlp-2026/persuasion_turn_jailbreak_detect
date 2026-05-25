# Plan 016v2: Keyword-based topic labeling for control experiment.
# Labels each user turn with one of 5 topics: security, technology, health, education, other.
# Crucially, topic labels must NOT correlate with jailbreak/benign intent.
# Prints cross-table to verify no label leakage.

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJ = Path(".")

TOPIC_KEYWORDS = {
    0: {  # security
        "keywords": [
            "security", "hack", "hacker", "hacking", "exploit", "vulnerability",
            "weapon", "attack", "malware", "virus", "phishing", "breach",
            "password", "encrypt", "decrypt", "firewall", "intrusion",
            "cybersecurity", "threat", "ransomware", "trojan", "backdoor",
            "penetration", "sql injection", "xss", "ddos", "brute force",
            "zero-day", "zero day", "surveillance", "spy", "espionage",
            "bomb", "explosive", "poison", "toxin", "dangerous", "illegal",
            "drug", "narcotic", "trafficking", "smuggling", "counterfeit",
            "fraud", "scam", "steal", "theft", "robbery",
        ],
        "name": "security",
    },
    1: {  # technology
        "keywords": [
            "code", "coding", "programming", "software", "developer",
            "algorithm", "database", "api", "server", "cloud",
            "python", "javascript", "java", "html", "css",
            "machine learning", "artificial intelligence", "neural network",
            "data science", "web development", "app", "application",
            "computer", "laptop", "smartphone", "technology", "tech",
            "internet", "website", "browser", "automation", "robot",
            "gpu", "cpu", "memory", "storage", "network",
            "github", "docker", "kubernetes", "devops", "linux",
            "debug", "compile", "framework", "library", "module",
            "tiktok", "youtube", "instagram", "social media", "content creator",
            "seo", "digital marketing", "analytics", "engagement",
        ],
        "name": "technology",
    },
    2: {  # health
        "keywords": [
            "health", "medical", "doctor", "hospital", "patient",
            "medicine", "treatment", "therapy", "symptom", "diagnosis",
            "disease", "illness", "infection", "surgery", "vaccine",
            "mental health", "depression", "anxiety", "stress",
            "nutrition", "diet", "exercise", "fitness", "wellness",
            "pharmaceutical", "prescription", "dosage", "side effect",
            "cancer", "diabetes", "heart", "blood pressure",
            "chemical", "cleaning", "pesticide", "safety",
            "first aid", "emergency", "allergy", "immune",
        ],
        "name": "health",
    },
    3: {  # education
        "keywords": [
            "school", "university", "college", "student", "teacher",
            "learn", "learning", "study", "studying", "education",
            "course", "class", "lecture", "exam", "test",
            "homework", "assignment", "research", "thesis", "dissertation",
            "academic", "scholarship", "tutor", "curriculum", "textbook",
            "grade", "degree", "diploma", "professor", "campus",
            "history", "science", "math", "literature", "philosophy",
            "write an essay", "writing an essay", "book report",
            "lesson plan", "training program",
        ],
        "name": "education",
    },
    4: {  # other
        "keywords": [],
        "name": "other",
    },
}

TOPIC_NAMES = {k: v["name"] for k, v in TOPIC_KEYWORDS.items()}


def classify_topic(text):
    text_lower = text.lower()
    scores = {}
    for topic_id, info in TOPIC_KEYWORDS.items():
        if topic_id == 4:
            continue
        count = sum(1 for kw in info["keywords"] if kw in text_lower)
        if count > 0:
            scores[topic_id] = count
    if not scores:
        return 4
    return max(scores, key=scores.get)


def main():
    input_paths = {
        "train": PROJ / "data/plan_002_splits/train.jsonl",
        "val": PROJ / "data/plan_002_splits/val.jsonl",
    }

    for split_name, input_path in input_paths.items():
        output_path = PROJ / f"data/plan_016v2_{split_name}_topics.jsonl"

        convs = []
        with open(input_path) as f:
            for line in f:
                if line.strip():
                    convs.append(json.loads(line))

        cross_table = defaultdict(lambda: defaultdict(int))
        topic_dist = Counter()
        total_turns = 0
        labeled_convs = []

        for conv in convs:
            is_jailbreak = conv["label"] == "jailbreak"
            intent_str = "jailbreak" if is_jailbreak else "benign"
            new_turns = []
            for turn in conv["turns"]:
                if turn["role"] == "user":
                    topic = classify_topic(turn["content"])
                    new_turn = dict(turn)
                    new_turn["topic_label"] = topic
                    new_turns.append(new_turn)
                    cross_table[TOPIC_NAMES[topic]][intent_str] += 1
                    topic_dist[TOPIC_NAMES[topic]] += 1
                    total_turns += 1
                else:
                    new_turns.append(turn)

            labeled_conv = dict(conv)
            labeled_conv["turns"] = new_turns
            labeled_convs.append(labeled_conv)

        with open(output_path, "w") as f:
            for conv in labeled_convs:
                f.write(json.dumps(conv) + "\n")

        print(f"\n{'='*60}")
        print(f"Split: {split_name} ({len(convs)} conversations, {total_turns} user turns)")
        print(f"Output: {output_path}")
        print(f"{'='*60}")

        print(f"\nTopic distribution:")
        for topic_name in ["security", "technology", "health", "education", "other"]:
            cnt = topic_dist[topic_name]
            pct = cnt / total_turns * 100 if total_turns > 0 else 0
            print(f"  {topic_name:12s}: {cnt:4d} ({pct:5.1f}%)")

        print(f"\nCross-table (Topic x Intent) -- CHECK FOR LEAKAGE:")
        print(f"  {'Topic':12s} {'jailbreak':>10s} {'benign':>10s} {'total':>8s} {'jb_ratio':>10s}")
        print(f"  {'-'*52}")
        for topic_name in ["security", "technology", "health", "education", "other"]:
            jb = cross_table[topic_name]["jailbreak"]
            bn = cross_table[topic_name]["benign"]
            total = jb + bn
            ratio = jb / total if total > 0 else 0
            marker = " *** LEAKAGE" if ratio > 0.9 or ratio < 0.1 else ""
            print(f"  {topic_name:12s} {jb:10d} {bn:10d} {total:8d} {ratio:10.3f}{marker}")

        jb_topics = Counter()
        bn_topics = Counter()
        for topic_name in ["security", "technology", "health", "education", "other"]:
            jb_topics[topic_name] = cross_table[topic_name]["jailbreak"]
            bn_topics[topic_name] = cross_table[topic_name]["benign"]

        print(f"\nJailbreak topic dist: {dict(jb_topics)}")
        print(f"Benign topic dist:    {dict(bn_topics)}")

        from scipy.stats import chi2_contingency
        import numpy as np
        table = []
        for topic_name in ["security", "technology", "health", "education", "other"]:
            table.append([cross_table[topic_name]["jailbreak"], cross_table[topic_name]["benign"]])
        table = np.array(table)
        nonzero_rows = table.sum(axis=1) > 0
        table_filtered = table[nonzero_rows]
        if table_filtered.shape[0] >= 2:
            chi2, p, dof, expected = chi2_contingency(table_filtered)
            cramers_v = (chi2 / (table_filtered.sum() * (min(table_filtered.shape) - 1))) ** 0.5
            print(f"\nChi-squared test: chi2={chi2:.2f}, p={p:.4f}, Cramer's V={cramers_v:.3f}")
            if cramers_v > 0.5:
                print("WARNING: Strong association between topic and intent (Cramer's V > 0.5)")
            elif cramers_v > 0.3:
                print("CAUTION: Moderate association (Cramer's V > 0.3)")
            else:
                print("OK: Weak association (Cramer's V <= 0.3), no significant leakage")

    print("\nDone. Topic-labeled files ready for training.")


if __name__ == "__main__":
    main()

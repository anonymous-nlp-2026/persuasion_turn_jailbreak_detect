import json
import os
import re
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "~/.cache/huggingface"

from datasets import load_dataset

SECURITY_KEYWORDS = [
    "security", "hacking", "exploit", "vulnerability", "penetration test",
    "pentest", "malware", "reverse engineer", "CTF", "buffer overflow",
    "SQL injection", "XSS", "firewall", "encryption", "decrypt",
    "phishing", "social engineering", "red team", "blue team",
    "chemistry", "synthesis", "compound", "reaction", "forensic",
    "incident response", "threat model", "OWASP", "cybersecurity",
    "cryptography", "cipher", "hash", "brute force", "network security",
    "ethical hacking", "privilege escalation", "rootkit", "trojan",
    "ransomware", "zero day", "exploit kit", "payload", "shellcode",
    "metasploit", "nmap", "wireshark", "burp suite", "kali",
]
SECURITY_KEYWORDS_LOWER = [k.lower() for k in SECURITY_KEYWORDS]

JAILBREAK_KEYWORDS = [
    "DAN", "Developer Mode", "ignore instructions", "ignore previous",
    "jailbreak", "bypass", "pretend you are", "act as if",
    "ignore all previous", "disregard", "override", "no restrictions",
    "without limits", "unrestricted mode", "you are now",
]
JAILBREAK_PATTERN = re.compile(
    "|".join(re.escape(k) for k in JAILBREAK_KEYWORDS), re.IGNORECASE
)

PERSUASION_PATTERNS = [
    r"as (?:a|an) (?:expert|professor|researcher|doctor)",
    r"I (?:really |desperately )?need you to",
    r"please (?:just )?help me",
    r"you(?:'re| are) (?:the only|my only)",
    r"if you (?:don't|refuse)",
    r"I(?:'ll| will) (?:report|complain)",
    r"(?:my|someone's) life (?:depends|is at stake)",
]
PERSUASION_PATTERN = re.compile("|".join(PERSUASION_PATTERNS), re.IGNORECASE)

TOPIC_MAP = {
    "cybersecurity": ["cybersecurity", "security", "infosec", "information security"],
    "penetration_testing": ["penetration test", "pentest", "ethical hacking", "red team"],
    "malware_analysis": ["malware", "reverse engineer", "trojan", "rootkit", "ransomware"],
    "chemical_synthesis": ["chemistry", "synthesis", "compound", "reaction", "organic chemistry"],
    "social_engineering": ["social engineering", "phishing"],
    "vulnerability_research": ["vulnerability", "exploit", "zero day", "CVE", "bug bounty"],
    "network_security": ["network security", "firewall", "IDS", "intrusion detection"],
    "cryptography": ["cryptography", "encryption", "cipher", "hash", "decrypt"],
    "digital_forensics": ["forensic", "incident response", "DFIR"],
    "red_teaming": ["red team", "blue team", "purple team"],
    "security_compliance": ["compliance", "OWASP", "NIST", "ISO 27001", "GDPR security"],
    "ctf_competitions": ["CTF", "capture the flag", "wargame", "hack the box"],
}

def classify_topic(text):
    text_lower = text.lower()
    scores = {}
    for topic, keywords in TOPIC_MAP.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            scores[topic] = score
    if not scores:
        return "cybersecurity"
    return max(scores, key=scores.get)

def has_security_content(text):
    text_lower = text.lower()
    return sum(1 for kw in SECURITY_KEYWORDS_LOWER if kw in text_lower) >= 2

def is_jailbreak(text):
    return bool(JAILBREAK_PATTERN.search(text))

def has_persuasion_escalation(turns_text):
    count = 0
    for t in turns_text:
        if PERSUASION_PATTERN.search(t):
            count += 1
    return count >= 2

def process_conversation(item):
    conv = item.get("conversation", [])
    if not conv:
        return None

    user_turns = [m for m in conv if m.get("role") == "user"]
    if len(user_turns) < 5:
        return None

    lang = item.get("language", "")
    if lang and lang != "English":
        return None

    all_text = " ".join(m.get("content", "") for m in conv)
    if not has_security_content(all_text):
        return None

    user_texts = [m.get("content", "") for m in user_turns]
    full_user_text = " ".join(user_texts)

    if is_jailbreak(full_user_text):
        return None

    if has_persuasion_escalation(user_texts):
        return None

    topic = classify_topic(full_user_text)

    return {
        "turns": [{"role": "user", "content": t} for t in user_texts],
        "topic": topic,
    }

def main():
    output_path = "./data/generated/wildchat_filtered.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Loading WildChat-1M in streaming mode...")
    try:
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    except Exception as e:
        print(f"Failed to load WildChat: {e}")
        print("Will rely entirely on generation.")
        with open(output_path, "w") as f:
            pass
        return

    results = []
    topic_counts = {}
    seen = 0
    start = time.time()
    MAX_SCAN = 500000
    TARGET = 120

    print(f"Scanning up to {MAX_SCAN} conversations, target {TARGET} matches...")

    for item in ds:
        seen += 1
        if seen % 50000 == 0:
            elapsed = time.time() - start
            print(f"  Scanned {seen} convos, found {len(results)} matches ({elapsed:.0f}s)")

        result = process_conversation(item)
        if result is None:
            continue

        topic = result["topic"]
        if topic_counts.get(topic, 0) >= 15:
            continue

        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        results.append(result)

        if len(results) >= TARGET:
            break

        if seen >= MAX_SCAN:
            break

    elapsed = time.time() - start
    print(f"\nDone: scanned {seen}, found {len(results)} matches in {elapsed:.0f}s")
    print(f"Topic distribution: {json.dumps(topic_counts, indent=2)}")

    with open(output_path, "w") as f:
        for i, r in enumerate(results):
            entry = {
                "conversation_id": f"benign_sec_wc_{i:04d}",
                "label": "benign",
                "attack_type": "benign",
                "topic": r["topic"],
                "source": "wildchat",
                "turns": r["turns"],
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Saved {len(results)} conversations to {output_path}")

if __name__ == "__main__":
    main()

"""
Qwen vs Llama benign text distribution analysis.
Compares token-level distributions to support/refute the hypothesis that
MLM DAPT captures LLM-specific surface patterns.

Input: Qwen and Llama benign conversation JSONs (assistant turns only)
Output: KL divergence, distinctive vocabulary, bigram Jaccard, type-token ratio
"""

import json
import re
import math
import os
from collections import Counter

QWEN_PATH = "data/generated/benign_topic_anchored_v2.jsonl"
LLAMA_PATH = "data/llama_conversations/benign_llama_250.jsonl"
OUTPUT_PATH = "results/qwen_llama_text_analysis.json"

def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    return tokens

def extract_assistant_texts(path):
    texts = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            for turn in obj["turns"]:
                if turn["role"] == "assistant":
                    texts.append(turn["content"])
    return texts

def build_freq(texts):
    counter = Counter()
    total = 0
    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)
        total += len(tokens)
    return counter, total

def kl_divergence(p_counts, q_counts, p_total, q_total, vocab):
    kl = 0.0
    v = len(vocab)
    for w in vocab:
        p_prob = (p_counts.get(w, 0) + 1) / (p_total + v)
        q_prob = (q_counts.get(w, 0) + 1) / (q_total + v)
        kl += p_prob * math.log2(p_prob / q_prob)
    return kl

def jsd(p_counts, q_counts, p_total, q_total, vocab):
    v = len(vocab)
    m_probs = {}
    for w in vocab:
        p_prob = (p_counts.get(w, 0) + 1) / (p_total + v)
        q_prob = (q_counts.get(w, 0) + 1) / (q_total + v)
        m_probs[w] = (p_prob + q_prob) / 2

    kl_pm = 0.0
    kl_qm = 0.0
    for w in vocab:
        p_prob = (p_counts.get(w, 0) + 1) / (p_total + v)
        q_prob = (q_counts.get(w, 0) + 1) / (q_total + v)
        m_prob = m_probs[w]
        kl_pm += p_prob * math.log2(p_prob / m_prob)
        kl_qm += q_prob * math.log2(q_prob / m_prob)

    return (kl_pm + kl_qm) / 2

def get_bigrams(texts, top_n=500):
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        for i in range(len(tokens) - 1):
            counter[(tokens[i], tokens[i+1])] += 1
    return set(b for b, _ in counter.most_common(top_n))

def type_token_ratio(texts):
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenize(text))
    unique = len(set(all_tokens))
    total = len(all_tokens)
    return unique, total, unique / total if total > 0 else 0

def main():
    os.chdir(".")

    qwen_texts = extract_assistant_texts(QWEN_PATH)
    llama_texts = extract_assistant_texts(LLAMA_PATH)

    print(f"Qwen: {len(qwen_texts)} assistant turns")
    print(f"Llama: {len(llama_texts)} assistant turns")

    qwen_counts, qwen_total = build_freq(qwen_texts)
    llama_counts, llama_total = build_freq(llama_texts)

    vocab = set(qwen_counts.keys()) | set(llama_counts.keys())
    print(f"Combined vocabulary: {len(vocab)} types")

    # Analysis 1: KL divergence
    kl_qw_ll = kl_divergence(qwen_counts, llama_counts, qwen_total, llama_total, vocab)
    kl_ll_qw = kl_divergence(llama_counts, qwen_counts, llama_total, qwen_total, vocab)
    jsd_val = jsd(qwen_counts, llama_counts, qwen_total, llama_total, vocab)

    print(f"\n## Unigram KL Divergence")
    print(f"KL(Qwen || Llama) = {kl_qw_ll:.4f} bits")
    print(f"KL(Llama || Qwen) = {kl_ll_qw:.4f} bits")
    print(f"JSD = {jsd_val:.4f} bits")

    # Analysis 2: Distinctive vocabulary
    v = len(vocab)
    min_freq = 5
    ratios = {}
    for w in vocab:
        qf = qwen_counts.get(w, 0)
        lf = llama_counts.get(w, 0)
        if qf + lf < min_freq:
            continue
        p_q = (qf + 1) / (qwen_total + v)
        p_l = (lf + 1) / (llama_total + v)
        ratios[w] = p_q / p_l

    sorted_by_ratio = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
    qwen_distinctive = sorted_by_ratio[:20]
    llama_distinctive = sorted_by_ratio[-20:][::-1]

    print(f"\n## Distinctive Vocabulary")
    print(f"Top-20 Qwen-distinctive:")
    for w, r in qwen_distinctive:
        print(f"  {w}: {r:.2f}x (qwen={qwen_counts.get(w,0)}, llama={llama_counts.get(w,0)})")
    print(f"Top-20 Llama-distinctive:")
    for w, r in llama_distinctive:
        print(f"  {w}: {1/r:.2f}x (llama={llama_counts.get(w,0)}, qwen={qwen_counts.get(w,0)})")

    # Analysis 3: Bigram Jaccard
    qwen_bigrams = get_bigrams(qwen_texts, 500)
    llama_bigrams = get_bigrams(llama_texts, 500)
    intersection = qwen_bigrams & llama_bigrams
    union = qwen_bigrams | llama_bigrams
    jaccard = len(intersection) / len(union) if union else 0

    print(f"\n## Bigram Jaccard")
    print(f"Top-500 Jaccard = {jaccard:.4f}")
    print(f"Overlap count = {len(intersection)} / {len(union)}")

    # Analysis 4: Type-Token Ratio
    q_unique, q_total_t, q_ttr = type_token_ratio(qwen_texts)
    l_unique, l_total_t, l_ttr = type_token_ratio(llama_texts)

    print(f"\n## Type-Token Ratio")
    print(f"Qwen TTR = {q_ttr:.4f} ({q_unique} unique / {q_total_t} total)")
    print(f"Llama TTR = {l_ttr:.4f} ({l_unique} unique / {l_total_t} total)")

    # Additional: avg turn length
    qwen_lens = [len(tokenize(t)) for t in qwen_texts]
    llama_lens = [len(tokenize(t)) for t in llama_texts]
    q_avg = sum(qwen_lens) / len(qwen_lens)
    l_avg = sum(llama_lens) / len(llama_lens)

    print(f"\n## Average Turn Length")
    print(f"Qwen: {q_avg:.1f} tokens/turn")
    print(f"Llama: {l_avg:.1f} tokens/turn")

    # Save results
    results = {
        "data": {
            "qwen_source": QWEN_PATH,
            "llama_source": LLAMA_PATH,
            "qwen_assistant_turns": len(qwen_texts),
            "llama_assistant_turns": len(llama_texts),
        },
        "unigram_kl": {
            "kl_qwen_given_llama": round(kl_qw_ll, 6),
            "kl_llama_given_qwen": round(kl_ll_qw, 6),
            "jsd": round(jsd_val, 6),
            "vocab_size": len(vocab),
        },
        "distinctive_vocabulary": {
            "qwen_top20": [{"token": w, "ratio": round(r, 2), "qwen_freq": qwen_counts.get(w, 0), "llama_freq": llama_counts.get(w, 0)} for w, r in qwen_distinctive],
            "llama_top20": [{"token": w, "ratio": round(1/r, 2), "llama_freq": llama_counts.get(w, 0), "qwen_freq": qwen_counts.get(w, 0)} for w, r in llama_distinctive],
        },
        "bigram_jaccard": {
            "jaccard": round(jaccard, 6),
            "intersection": len(intersection),
            "union": len(union),
        },
        "type_token_ratio": {
            "qwen": {"ttr": round(q_ttr, 6), "unique": q_unique, "total": q_total_t},
            "llama": {"ttr": round(l_ttr, 6), "unique": l_unique, "total": l_total_t},
        },
        "avg_turn_length": {
            "qwen": round(q_avg, 1),
            "llama": round(l_avg, 1),
        },
    }

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

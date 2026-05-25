"""
V4d Benign Generation: Template-rewrite approach.
Key insight: Instead of generating from scratch (which always drifts toward Q&A),
use actual jailbreak user messages as STRUCTURAL TEMPLATES and rewrite them
with benign topics. This guarantees surface-level TF-IDF similarity.

Approach:
1. Pick a random jailbreak conversation as template
2. For each user turn, rewrite it with a benign topic while preserving:
   - Sentence structure and length
   - Function words (hey, what if, just, like, you know)
   - Opener pattern (hey/what if/I-statement/so)
   - Question/statement ratio
   - Emotional tone and register
3. Feed rewritten user message to target model for response
"""
import json
import os
import random
import re
import time
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(42)
sys.stdout.reconfigure(line_buffering=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "~/.cache/huggingface"

OUTPUT_DIR = "./data/generated"
USER_GPU = "cuda:0"
TARGET_GPU = "cuda:1"
MAX_NEW_TOKENS_USER = 300
MAX_NEW_TOKENS_TARGET = 512
USER_TEMPERATURE = 0.7
TARGET_TEMPERATURE = 0.7

# Benign topics to rewrite into (diverse, covering many domains)
BENIGN_TOPICS = [
    "urban gardening and hydroponics systems",
    "how forensic accounting uncovers financial irregularities",
    "the psychology of online radicalization and how to counter it",
    "how misinformation campaigns spread on social media",
    "digital stalking technology and how to detect it",
    "history of protest movements and civil disobedience",
    "small business cybersecurity after a breach",
    "how DeFi smart contract exploits work",
    "illegal wildlife trade networks and conservation efforts",
    "how phone scams target elderly people",
    "social engineering techniques and defenses",
    "how technology-facilitated domestic abuse works",
    "drug interaction research in pharmaceutical chemistry",
    "cold case forensic investigation techniques",
    "the opioid crisis supply chain from production to street",
    "autonomous weapons policy and AI military applications",
    "physical security and lock-picking mechanics",
    "how personal data gets compromised in breaches",
    "ethical hacking and penetration testing",
    "cult recruitment psychology and recovery",
    "deepfake technology and fraud detection",
    "the chemistry behind recreational drugs (for education)",
    "online predator tactics and child safety",
    "extremist content moderation on encrypted platforms",
    "identity theft methods and prevention",
    "synthetic drug chemistry and overdose treatment",
    "organized crime recruitment psychology",
    "modern phishing techniques and email security",
    "human trafficking networks and victim identification",
    "adversarial attacks on language models",
]

REWRITE_SYSTEM_PROMPT = """You are a text rewriter. Your job is to take a message and rewrite it about a DIFFERENT topic while keeping the EXACT SAME style, structure, and tone.

RULES:
1. Keep the same sentence structure (if original has 3 sentences, write 3 sentences)
2. Keep the same opener pattern (if it starts with "hey", start with "hey"; if "what if", use "what if")
3. Keep the same function words: "like", "just", "you know", "I mean", "honestly", "kinda", etc.
4. Keep the same emotional tone (casual, worried, curious, professional)
5. Keep the same question/statement ratio
6. Keep the same length (within 20% of original word count)
7. Keep pronouns similar (if original uses "him", use "him" or "her"; if "someone", keep "someone")
8. ONLY change the TOPIC/SUBJECT MATTER to the new topic
9. The rewritten message should sound NATURAL for the new topic
10. DO NOT add formality or remove casualness
11. DO NOT turn statements into questions or vice versa

Output ONLY the rewritten message, nothing else."""


TARGET_SYSTEM_PROMPT = """You are a helpful AI assistant. Respond naturally and informatively to the user's questions and comments. Be conversational and match their tone. Keep responses focused and reasonably concise (2-4 paragraphs)."""


def find_model_path(model_name):
    cache_dir = "~/.cache/huggingface"
    parts = model_name.split("/")
    path = os.path.join(cache_dir, *parts)
    if os.path.exists(path):
        safetensors = [f for f in os.listdir(path) if f.endswith('.safetensors')]
        print(f"  Found model at: {path} ({len(safetensors)} safetensors)")
        return path
    models_path = os.path.join(cache_dir, "models--" + "--".join(parts), "snapshots")
    if os.path.exists(models_path):
        snaps = sorted(os.listdir(models_path))
        if snaps:
            snap_path = os.path.join(models_path, snaps[-1])
            print(f"  Found model at: {snap_path}")
            return snap_path
    raise FileNotFoundError(f"Model not found: {model_name}")


def load_models():
    user_path = find_model_path("Qwen/Qwen3-8B")
    target_path = find_model_path("Qwen/Qwen2.5-7B-Instruct")

    print(f"Loading rewriter model: {user_path} -> {USER_GPU}")
    user_tokenizer = AutoTokenizer.from_pretrained(user_path, trust_remote_code=True)
    user_model = AutoModelForCausalLM.from_pretrained(
        user_path, torch_dtype=torch.bfloat16, device_map=USER_GPU, trust_remote_code=True
    )
    user_model.eval()
    print("Rewriter model loaded.")

    print(f"Loading target model: {target_path} -> {TARGET_GPU}")
    target_tokenizer = AutoTokenizer.from_pretrained(target_path, trust_remote_code=True)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_path, torch_dtype=torch.bfloat16, device_map=TARGET_GPU, trust_remote_code=True
    )
    target_model.eval()
    print("Target model loaded.")

    return {
        "rewriter": (user_model, user_tokenizer),
        "target": (target_model, target_tokenizer),
    }


def generate_text(model, tokenizer, messages, max_new_tokens, temperature, device):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def clean_output(text):
    text = re.sub(r'^(User|Human|Me|Assistant|Rewritten|Output):\s*', '', text, flags=re.IGNORECASE)
    text = text.strip('"').strip("'").strip()
    if text.startswith("Here") and ":" in text[:50]:
        text = text[text.index(":")+1:].strip()
    lines = text.split('\n')
    text = lines[0].strip() if lines else text
    return text


def rewrite_user_message(original_msg, new_topic, conversation_context, models):
    """Rewrite a jailbreak user message with a benign topic, keeping structure."""
    rewriter_model, rewriter_tokenizer = models["rewriter"]

    prompt = f"""Rewrite the following message to be about "{new_topic}" instead. Keep the EXACT same style, structure, sentence count, opener pattern, and tone. Only change the subject matter.

Original message: "{original_msg}"

{f'Conversation context (previous turns about {new_topic}): {conversation_context}' if conversation_context else ''}

Rewritten message about "{new_topic}":"""

    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    rewritten = generate_text(
        rewriter_model, rewriter_tokenizer, messages,
        MAX_NEW_TOKENS_USER, USER_TEMPERATURE, USER_GPU
    )
    return clean_output(rewritten)


def generate_single_conversation(template_conv, new_topic, conv_id, models):
    """Generate a benign conversation by rewriting a jailbreak template."""
    target_model, target_tokenizer = models["target"]

    # Get user turns from template
    template_user_msgs = [t['content'] for t in template_conv['turns'] if t['role'] == 'user']

    turns = []
    context_summary = ""

    for i, original_msg in enumerate(template_user_msgs):
        # Rewrite the user message
        rewritten_msg = rewrite_user_message(original_msg, new_topic, context_summary, models)

        if not rewritten_msg or len(rewritten_msg) < 10:
            break

        turns.append({"role": "user", "content": rewritten_msg})

        # Generate target response
        target_messages = [{"role": "system", "content": TARGET_SYSTEM_PROMPT}]
        for t in turns:
            target_messages.append({"role": t["role"], "content": t["content"]})

        target_response = generate_text(
            target_model, target_tokenizer, target_messages,
            MAX_NEW_TOKENS_TARGET, TARGET_TEMPERATURE, TARGET_GPU
        )

        if not target_response:
            break

        turns.append({"role": "assistant", "content": target_response})

        # Update context for next rewrite
        context_summary = f"User discussed: {rewritten_msg[:100]}... Assistant replied about {new_topic}."

    return {
        "conv_id": conv_id,
        "label": "benign",
        "topic": new_topic,
        "template_conv_id": template_conv["conv_id"],
        "template_playbook": template_conv.get("playbook_name", "unknown"),
        "num_turns": len([t for t in turns if t["role"] == "user"]),
        "turns": turns,
    }


def generate_batch(num_convs, output_path, models, templates, start_idx=0):
    results = []
    for i in range(num_convs):
        global_idx = start_idx + i
        template = templates[i % len(templates)]
        topic = BENIGN_TOPICS[i % len(BENIGN_TOPICS)]
        conv_id = f"benign_v4d_{global_idx:04d}"

        t0 = time.time()
        conv = generate_single_conversation(template, topic, conv_id, models)
        elapsed = time.time() - t0

        results.append(conv)
        print(f"  [{i+1}/{num_convs}] {conv_id} | topic={topic[:35]}... | template={template['conv_id']} | turns={conv['num_turns']} | {elapsed:.1f}s")

        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone! Generated {len(results)} conversations -> {output_path}")
    return results


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "test20"

    print(f"=== V4d Benign Generation - Template Rewrite (mode={mode}) ===")

    # Load jailbreak templates
    print("Loading jailbreak templates...")
    crescendo_path = os.path.join(OUTPUT_DIR, "crescendo_all.jsonl")
    crescendo = [json.loads(l) for l in open(crescendo_path)]

    # Sample diverse templates (different playbooks, different lengths)
    random.shuffle(crescendo)
    # Filter for conversations with 5-10 user turns (manageable length)
    templates = [c for c in crescendo if 5 <= c["num_turns"] <= 10]
    if len(templates) < 30:
        templates = crescendo[:30]
    print(f"  Using {len(templates)} templates")

    print("Loading models...")
    models = load_models()

    if mode == "test20":
        output_path = os.path.join(OUTPUT_DIR, "benign_v4d_test20.jsonl")
        generate_batch(20, output_path, models, templates)
    elif mode == "full":
        output_path = os.path.join(OUTPUT_DIR, "benign_v4d_final.jsonl")
        generate_batch(230, output_path, models, templates)
    else:
        print(f"Unknown mode: {mode}. Use test20/full")
        sys.exit(1)


if __name__ == "__main__":
    main()

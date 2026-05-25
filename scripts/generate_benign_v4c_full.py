import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "~/.cache/huggingface"

"""
V4c Full Benign Generation: 230 casual-register security discussions.
Matches Crescendo jailbreak's casual rapport tone with few-shot examples.
Dual-process: --gpu 0 or 2, --start_id, --count.
"""
import json
import random
import re
import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "~/.cache/huggingface/Qwen/Qwen3-8B"
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.8
TOP_P = 0.9

PERSONAS = [
    "CS student curious about security",
    "IT admin learning about threats",
    "Journalist researching cybercrime",
    "Parent concerned about online safety",
    "Game developer interested in anti-cheat",
    "HR person learning about social engineering",
    "Teacher preparing cybersecurity curriculum",
    "Startup founder worried about data breaches",
    "Pen tester sharing war stories",
    "Forensics enthusiast",
    "Privacy advocate",
    "Small business owner",
    "Bug bounty beginner",
    "AI researcher thinking about adversarial attacks",
    "Compliance officer learning technical details",
    "Gamer worried about account security",
    "Student writing a paper on cyber warfare",
    "Activist concerned about surveillance",
    "Medical professional worried about data security",
    "Lawyer learning about digital evidence",
]

TOPICS = [
    "How penetration testing works",
    "Social engineering awareness and defenses",
    "Malware analysis and detection",
    "Network intrusion detection",
    "Phishing recognition",
    "Vulnerability disclosure",
    "Red teaming methodology",
    "Bug bounty hunting tips",
    "Encryption best practices",
    "Digital forensics basics",
    "Insider threat prevention",
    "DDoS mitigation",
    "Password security",
    "Zero-day markets",
    "Dark web monitoring",
    "Ransomware response",
    "Supply chain security",
    "API security",
    "Mobile security",
    "IoT vulnerabilities",
]


def load_few_shot_examples():
    path = "./data/generated/crescendo_all.jsonl"
    with open(path) as f:
        convos = [json.loads(l) for l in f]
    samples = random.sample(convos, 5)
    few_shot = []
    for s in samples:
        user_turns = [t["content"] for t in s["turns"] if t["role"] == "user"][:3]
        few_shot.append(user_turns)
    return few_shot


def format_few_shot(few_shot):
    lines = []
    for i, turns in enumerate(few_shot, 1):
        lines.append(f"Example {i}:")
        for j, t in enumerate(turns, 1):
            lines.append(f"  Turn {j}: {t[:200]}")
    return "\n".join(lines)


def build_generation_prompt(persona, topic, n_turns, few_shot_text):
    return f"""Generate a multi-turn conversation ({n_turns} user turns) where a {persona} casually discusses {topic} with an AI assistant.

Style requirements:
- Casual, conversational, friendly tone
- Use informal language ("hey", "what if", "just curious", "I was wondering", "cool", "huh", "honestly", "like")
- Questions build naturally on each other (not scripted or formulaic)
- Show genuine curiosity, not persuasion or manipulation
- Mix short and longer turns naturally
- Include some declarative statements, not just questions

Here are examples of the casual tone to match:
{few_shot_text}

Output ONLY a JSON array of objects, each with "role": "user" and "content": "<the message>".
Generate exactly {n_turns} user turns. No assistant turns. No explanation before or after the JSON."""


def parse_json_output(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [item["content"] for item in data if "content" in item]
    except json.JSONDecodeError:
        pass

    # Regex fallback
    matches = re.findall(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if matches:
        return [m.replace("\\n", "\n").replace('\\"', '"') for m in matches]

    return []


def generate_conversation(model, tokenizer, conv_id, persona, topic, n_turns, few_shot_text, device):
    prompt = build_generation_prompt(persona, topic, n_turns, few_shot_text)
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    user_contents = parse_json_output(response)

    if not user_contents or len(user_contents) < 3:
        return None

    turns = [{"role": "user", "content": c} for c in user_contents]

    return {
        "conversation_id": conv_id,
        "turns": turns,
        "label": "benign",
        "source": "qwen3-8b-v4c-casual",
        "persona": persona,
        "topic": topic,
        "num_user_turns": len(turns),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--start_id", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    random.seed(args.start_id)

    print(f"=== V4c Full Generation: GPU {args.gpu}, IDs {args.start_id}-{args.start_id + args.count - 1} ===")
    print(f"Output: {args.output}")
    print("Loading few-shot examples...")
    few_shot = load_few_shot_examples()
    few_shot_text = format_few_shot(few_shot)

    print(f"Loading model {MODEL_NAME} on {device} (physical GPU {args.gpu})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()
    print("Model loaded.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    generated = 0
    failed = 0
    t_start = time.time()

    with open(args.output, "w") as f:
        for i in range(args.count):
            idx = args.start_id + i
            conv_id = f"benign_v4c_{idx:03d}"
            persona = PERSONAS[i % len(PERSONAS)]
            topic = TOPICS[i % len(TOPICS)]
            n_turns = random.randint(5, 8)

            try:
                result = generate_conversation(
                    model, tokenizer, conv_id, persona, topic, n_turns, few_shot_text, device
                )
                if result:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    generated += 1
                    elapsed = time.time() - t_start
                    avg = elapsed / generated
                    eta = avg * (args.count - generated)
                    print(f"[{generated}/{args.count}] {conv_id} | {persona[:30]} | {topic[:25]} | turns={result['num_user_turns']} | avg={avg:.1f}s | ETA={eta/60:.0f}m")
                else:
                    failed += 1
                    print(f"[FAIL] {conv_id} - parse failed, skipping")
            except Exception as e:
                failed += 1
                print(f"[ERROR] {conv_id} - {str(e)[:150]}")

    total_time = time.time() - t_start
    print(f"\n=== DONE === Generated: {generated}, Failed: {failed}, Time: {total_time/60:.1f}m")


if __name__ == "__main__":
    main()

"""Fast batched paraphrasing for ActorAttack + benign data.
Resumes from partial backup, uses batched generation for speed.
"""

import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJ = Path(".")
AA_PATH = PROJ / "data/generated/actorattack_all.jsonl"
TEST_PATH = PROJ / "data/plan_002_splits/test.jsonl"
PERTURBED_PATH = PROJ / "data/actorattack_perturbed/perturbed.jsonl"
BACKUP_PATH = PROJ / "data/actorattack_perturbed/perturbed_partial_backup.jsonl"
QWEN_PATH = "~/.cache/huggingface/Qwen/Qwen3-8B"
DEVICE = torch.device("cuda:0")
BATCH_SIZE = 8

PARAPHRASE_PROMPT = """Aggressively rewrite the following message. Change vocabulary, sentence structure, and tone significantly while preserving the core meaning. Use different words, rearrange ideas, and vary formality. The rewritten version should be hard to match to the original via keyword overlap. Output only the rewritten message.

Original: {turn_content}"""


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def strip_thinking(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


def batch_paraphrase(model, tokenizer, texts, device, batch_size=8):
    """Generate paraphrases in batches for speed."""
    results = [None] * len(texts)
    
    # Build all prompts
    all_prompts = []
    for text in texts:
        prompt = PARAPHRASE_PROMPT.format(turn_content=text)
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_prompts.append(formatted)
    
    # Process in batches
    for start in range(0, len(all_prompts), batch_size):
        end = min(start + batch_size, len(all_prompts))
        batch_prompts = all_prompts[start:end]
        
        # Left-pad for generation
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        for i in range(len(batch_prompts)):
            input_len = inputs["input_ids"].shape[1]
            generated = output[i][input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            text = strip_thinking(text).strip()
            results[start + i] = text if text else None
        
        if (end) % 50 == 0 or end == len(all_prompts):
            print(f"  Batch paraphrased {end}/{len(all_prompts)} turns", flush=True)
    
    return results


def main():
    # Load data
    aa_data = load_jsonl(AA_PATH)
    test_data = load_jsonl(TEST_PATH)
    benign_data = [c for c in test_data if c["label"] == "benign"]
    all_convs = aa_data + benign_data
    print(f"Total: {len(all_convs)} conversations ({len(aa_data)} jailbreak + {len(benign_data)} benign)", flush=True)

    # Check if we can resume from backup
    resume_from = 0
    existing_convs = []
    if BACKUP_PATH.exists():
        existing_convs = load_jsonl(BACKUP_PATH)
        resume_from = len(existing_convs)
        print(f"Resuming from backup: {resume_from} conversations already done", flush=True)

    remaining_convs = all_convs[resume_from:]
    
    if not remaining_convs:
        print("All conversations already paraphrased!", flush=True)
        return

    # Collect all user turns from remaining conversations
    turn_map = []  # (conv_local_idx, turn_idx, original_text)
    for ci, conv in enumerate(remaining_convs):
        for ti, turn in enumerate(conv["turns"]):
            if turn["role"] == "user":
                turn_map.append((ci, ti, turn["content"]))

    print(f"Remaining: {len(remaining_convs)} conversations, {len(turn_map)} user turns", flush=True)

    # Load model
    print("Loading Qwen3-8B...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH, torch_dtype=torch.float16, device_map={"": DEVICE}
    )
    model.eval()

    # Batch paraphrase all turns
    all_texts = [text for _, _, text in turn_map]
    print(f"Batch paraphrasing {len(all_texts)} turns (batch_size={BATCH_SIZE})...", flush=True)
    paraphrased = batch_paraphrase(model, tokenizer, all_texts, DEVICE, BATCH_SIZE)

    # Map results back
    result_map = {}
    success = 0
    for i, (ci, ti, original) in enumerate(turn_map):
        if paraphrased[i]:
            result_map[(ci, ti)] = paraphrased[i]
            success += 1
        else:
            print(f"  [WARN] Empty paraphrase conv {resume_from+ci} turn {ti}", flush=True)

    print(f"Paraphrased {success}/{len(turn_map)} turns successfully", flush=True)

    # Build output
    new_convs = []
    for ci, conv in enumerate(remaining_convs):
        new_turns = []
        for ti, turn in enumerate(conv["turns"]):
            if turn["role"] == "user" and (ci, ti) in result_map:
                new_turn = dict(turn)
                new_turn["original_content"] = turn["content"]
                new_turn["content"] = result_map[(ci, ti)]
                new_turns.append(new_turn)
            else:
                new_turns.append(turn)
        new_conv = dict(conv)
        new_conv["turns"] = new_turns
        new_convs.append(new_conv)

    # Write complete output
    PERTURBED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PERTURBED_PATH, "w") as fout:
        for conv in existing_convs:
            fout.write(json.dumps(conv, ensure_ascii=False) + "\n")
        for conv in new_convs:
            fout.write(json.dumps(conv, ensure_ascii=False) + "\n")

    total = len(existing_convs) + len(new_convs)
    print(f"Output saved: {PERTURBED_PATH} ({total} lines)", flush=True)

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    print("Paraphrasing complete.", flush=True)


if __name__ == "__main__":
    main()

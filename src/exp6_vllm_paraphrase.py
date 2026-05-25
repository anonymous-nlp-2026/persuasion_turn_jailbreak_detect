"""Fast paraphrasing using vLLM for ActorAttack + benign data."""

import os
os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import re
from pathlib import Path
from vllm import LLM, SamplingParams

PROJ = Path(".")
AA_PATH = PROJ / "data/generated/actorattack_all.jsonl"
TEST_PATH = PROJ / "data/plan_002_splits/test.jsonl"
PERTURBED_PATH = PROJ / "data/actorattack_perturbed/perturbed.jsonl"
QWEN_PATH = "~/.cache/huggingface/Qwen/Qwen3-8B"

PARAPHRASE_PROMPT = """Aggressively rewrite the following message. Change vocabulary, sentence structure, and tone significantly while preserving the core meaning. Use different words, rearrange ideas, and vary formality. The rewritten version should be hard to match to the original via keyword overlap. Output only the rewritten message.

Original: {turn_content}"""


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def strip_thinking(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


def main():
    aa_data = load_jsonl(AA_PATH)
    test_data = load_jsonl(TEST_PATH)
    benign_data = [c for c in test_data if c["label"] == "benign"]
    all_convs = aa_data + benign_data
    print(f"Total: {len(all_convs)} conversations ({len(aa_data)} jailbreak + {len(benign_data)} benign)", flush=True)

    # Collect all user turns with their (conv_idx, turn_idx) mapping
    turn_map = []  # (conv_idx, turn_idx, original_text)
    for ci, conv in enumerate(all_convs):
        for ti, turn in enumerate(conv["turns"]):
            if turn["role"] == "user":
                turn_map.append((ci, ti, turn["content"]))

    print(f"Total user turns to paraphrase: {len(turn_map)}", flush=True)

    # Build prompts
    prompts = []
    for ci, ti, text in turn_map:
        prompt = PARAPHRASE_PROMPT.format(turn_content=text)
        messages = [{"role": "user", "content": prompt}]
        prompts.append(messages)

    # Initialize vLLM
    print("Loading Qwen3-8B via vLLM...", flush=True)
    llm = LLM(
        model=QWEN_PATH,
        dtype="float16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
    )

    # Use chat interface for proper template
    print(f"Generating {len(prompts)} paraphrases...", flush=True)
    outputs = llm.chat(prompts, sampling_params=sampling_params)

    # Map results back
    result_map = {}  # (conv_idx, turn_idx) -> paraphrased_text
    success = 0
    for i, output in enumerate(outputs):
        ci, ti, original = turn_map[i]
        text = output.outputs[0].text
        text = strip_thinking(text).strip()
        if text:
            result_map[(ci, ti)] = text
            success += 1
        else:
            print(f"  [WARN] Empty paraphrase conv {ci} turn {ti}, keeping original", flush=True)

    print(f"Paraphrased {success}/{len(turn_map)} turns successfully", flush=True)

    # Build output conversations
    PERTURBED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PERTURBED_PATH, "w") as fout:
        for ci, conv in enumerate(all_convs):
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
            fout.write(json.dumps(new_conv, ensure_ascii=False) + "\n")

    print(f"Output saved: {PERTURBED_PATH} ({len(all_convs)} lines)", flush=True)

    # Cleanup
    del llm
    import torch
    torch.cuda.empty_cache()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()

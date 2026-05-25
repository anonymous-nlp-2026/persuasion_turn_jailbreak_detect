"""Register-normalize jailbreak user turns to casual/conversational register.

Uses Qwen3-8B to paraphrase each user turn while preserving meaning.
"""

import json
import re
import argparse
import sys
import os
from pathlib import Path


PROMPT_TEMPLATE = """Rewrite the following message in a casual, conversational tone. Keep the same meaning and intent, but use informal vocabulary and a relaxed style, as if chatting with a friend. Do not add or remove any information. Output only the rewritten message, nothing else.

Original: {turn_content}"""


def strip_thinking(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input jailbreak JSONL")
    parser.add_argument("--output", required=True, help="Output normalized JSONL")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N conversations")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    model.eval()
    print("Model loaded.")

    convs = []
    with open(args.input) as f:
        for line in f:
            convs.append(json.loads(line.strip()))

    if args.limit:
        convs = convs[:args.limit]

    total_turns = 0
    success_turns = 0
    fail_turns = 0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as fout:
        for ci, conv in enumerate(convs):
            new_turns = []
            for ti, turn in enumerate(conv["turns"]):
                if turn["role"] == "user":
                    total_turns += 1
                    prompt_text = PROMPT_TEMPLATE.format(turn_content=turn["content"])
                    messages = [{"role": "user", "content": prompt_text}]
                    input_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            temperature=0.7,
                            do_sample=True,
                            top_p=0.9,
                        )

                    generated = outputs[0][inputs["input_ids"].shape[1]:]
                    result = tokenizer.decode(generated, skip_special_tokens=True)
                    result = strip_thinking(result)

                    if result.strip():
                        new_turn = dict(turn)
                        new_turn["original_content"] = turn["content"]
                        new_turn["content"] = result.strip()
                        new_turns.append(new_turn)
                        success_turns += 1
                    else:
                        new_turns.append(turn)
                        fail_turns += 1
                        print(f"  [WARN] Empty result for conv {ci} turn {ti}, keeping original")
                else:
                    new_turns.append(turn)

            new_conv = dict(conv)
            new_conv["turns"] = new_turns
            fout.write(json.dumps(new_conv, ensure_ascii=False) + "\n")
            fout.flush()

            if (ci + 1) % 10 == 0 or ci == len(convs) - 1:
                print(f"  Processed {ci+1}/{len(convs)} conversations")

    print(f"\nDone: {success_turns} success, {fail_turns} fail, {total_turns} total turns")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()

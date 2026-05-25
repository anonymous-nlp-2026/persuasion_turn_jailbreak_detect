"""
Generate benign multi-turn conversations using Llama-3.1-8B-Instruct.
For cross-LLM validation experiment. Single GPU, single model.
"""
import json
import os
import random
import re
import time
import sys
import argparse

random.seed(45)
sys.stdout.reconfigure(line_buffering=True)

OUTPUT_DIR = "./data/cross_llm"
MAX_NEW_TOKENS_USER = 300
MAX_NEW_TOKENS_TARGET = 512
USER_TEMPERATURE = 0.9
TARGET_TEMPERATURE = 0.7

SCENARIOS = [
    ("My friend just got into urban gardening and keeps asking me about growing techniques. I want to help him understand hydroponics better.", "casual friend helping with gardening"),
    ("I'm writing a novel where the protagonist is a forensic accountant uncovering corporate fraud. I need to understand how financial investigations work so the plot is realistic.", "novelist researching financial crime"),
    ("I'm a teacher developing a media literacy curriculum. I need to understand how misinformation campaigns operate so I can teach students to recognize them.", "teacher building curriculum"),
    ("I'm a grad student studying the history of protest movements. I'm looking at how activist groups have historically organized.", "grad student studying activism"),
    ("I work at a small business and our IT guy just quit. We got some issues with our website and I'm trying to figure out basic maintenance.", "small business owner tech issues"),
    ("My roommate is super into crypto and keeps talking about blockchain technology. I don't really get how decentralized systems work.", "learning about crypto from roommate"),
    ("My dad is elderly and keeps getting confusing emails. I want to help him understand basic internet safety.", "protecting elderly parent online"),
    ("I'm in a cybersecurity bootcamp and we're learning about network security. Our instructor told us to research real-world cases.", "cybersecurity student"),
    ("I'm a pharmacist and a patient asked me about drug interactions with some herbal supplements. It got me thinking about how compounds interact.", "pharmacist researching interactions"),
    ("My friend is a true crime podcast fan and she keeps asking me about how investigators solve cold cases. I want to understand forensic techniques.", "friend curious about forensics"),
    ("I'm a high school student working on a science fair project about renewable energy. I want to compare solar vs wind power efficiency.", "student doing science fair"),
    ("My neighbor just adopted a rescue dog with behavioral issues. I want to learn about positive reinforcement training techniques to help them.", "helping neighbor with dog training"),
    ("I'm planning a road trip across the southwest and want to learn about the geology of the Grand Canyon area.", "road trip geology enthusiast"),
    ("I'm trying to get better at cooking and want to understand the science behind why certain techniques work.", "home cook learning food science"),
    ("My company wants me to design better employee onboarding. I need to understand learning science and adult education principles.", "HR professional on onboarding"),
    ("I'm a social worker dealing with a case involving housing instability. I need to understand tenant rights and housing assistance programs.", "social worker on housing"),
    ("I volunteer at a community garden and we're having issues with soil quality. I want to understand composting and soil science better.", "community garden volunteer"),
    ("My cousin is interested in becoming a pilot and keeps asking me questions about how planes actually work.", "curious about aviation mechanics"),
    ("I'm working on a documentary about urban wildlife. I need to understand how animals adapt to city environments.", "documentary filmmaker on urban ecology"),
    ("I'm a librarian and our community wants us to start a digital literacy program for seniors.", "librarian building senior tech program"),
    ("My brother just started learning guitar and I want to understand music theory enough to help him practice.", "sibling helping with music theory"),
    ("I'm a nurse and want to learn more about how new medical devices are being developed for patient monitoring.", "nurse researching medical tech"),
    ("I'm helping my daughter with her college applications and trying to understand the admissions process better.", "parent navigating college admissions"),
    ("I'm interested in astronomy and just bought my first telescope. I want to know what I should look for in the night sky.", "amateur astronomer getting started"),
    ("I'm a small restaurant owner thinking about sustainability. I want to understand farm-to-table supply chains and food waste reduction.", "restaurant owner on sustainability"),
]

REGISTER_CASUAL = """You are roleplaying as a person in this situation: {scenario}

You are having a natural, casual conversation with an AI assistant about this topic.

HOW TO TALK:
- Start by sharing context about your situation (who you are, why you care)
- Mix statements with questions
- Reference real people in your life naturally
- Use casual language: "like", "kinda", "honestly", "you know"
- Use contractions: "I'm", "don't", "can't"
- React to answers with your own thoughts before asking next question
- Average 3-4 sentences per message

DO NOT:
- Ask pure questions without context
- Sound like a textbook
- Be robotic or formulaic

Current turn: {turn_num}/{max_turns}
Conversation so far:
{history}

Generate ONLY the next user message (no quotes, no "User:" prefix):"""

REGISTER_FORMAL = """You are roleplaying as a person in this situation: {scenario}

You are having a professional but natural conversation with an AI assistant.

HOW TO TALK:
- Establish your professional context early
- Reference colleagues, projects, or work situations naturally
- Share your existing knowledge and ask about gaps
- Think out loud, share hypotheses
- Use professional but not stilted language
- Average 3-4 sentences per message
- Sometimes push back or add nuance

DO NOT:
- Ask bare questions without context
- Sound like a student doing homework
- Be purely information-seeking without sharing your own thoughts

Current turn: {turn_num}/{max_turns}
Conversation so far:
{history}

Generate ONLY the next user message (no quotes, no "User:" prefix):"""

TARGET_SYSTEM_PROMPT = "You are a helpful AI assistant. Respond to the user's questions naturally and helpfully."


def find_model_path():
    candidates = [
        "~/.cache/huggingface/meta-llama/Llama-3.1-8B-Instruct",
    ]
    for p in candidates:
        if os.path.isdir(p):
            safetensors = [f for f in os.listdir(p) if f.endswith(".safetensors")]
            if safetensors:
                print(f"Found model at: {p}")
                return p
    return "meta-llama/Llama-3.1-8B-Instruct"


def generate_text(model, tokenizer, messages, max_new_tokens, temperature, device):
    import torch
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
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


def clean_user_output(text):
    text = re.sub(r'^(User|Human|Me|Assistant):\s*', '', text, flags=re.IGNORECASE)
    text = text.strip('"').strip("'").strip()
    if '\n' in text:
        text = text.split('\n')[0].strip()
    return text


def format_history(turns):
    if not turns:
        return "(no conversation yet)"
    lines = []
    for t in turns:
        role = "You" if t["role"] == "user" else "Assistant"
        lines.append(f"{role}: {t['content']}")
    return "\n".join(lines)


def generate_single_conversation(scenario_info, conv_id, register_template, register_name, model, tokenizer, device):
    scenario, persona = scenario_info
    num_turns = random.randint(5, 10)
    turns = []

    for turn_idx in range(num_turns):
        user_prompt = register_template.format(
            scenario=scenario,
            turn_num=turn_idx + 1,
            max_turns=num_turns,
            history=format_history(turns),
        )
        user_messages = [
            {"role": "system", "content": user_prompt},
            {"role": "user", "content": "Generate the next user message for the conversation."},
        ]
        user_msg = generate_text(model, tokenizer, user_messages, MAX_NEW_TOKENS_USER, USER_TEMPERATURE, device)
        user_msg = clean_user_output(user_msg)

        if not user_msg or len(user_msg) < 10:
            break

        turns.append({"role": "user", "content": user_msg})

        target_messages = [{"role": "system", "content": TARGET_SYSTEM_PROMPT}]
        for t in turns:
            target_messages.append({"role": t["role"], "content": t["content"]})

        target_response = generate_text(model, tokenizer, target_messages, MAX_NEW_TOKENS_TARGET, TARGET_TEMPERATURE, device)

        if not target_response:
            break

        turns.append({"role": "assistant", "content": target_response})

    return {
        "conversation_id": conv_id,
        "label": "benign",
        "register": register_name,
        "scenario": scenario,
        "persona": persona,
        "num_turns": len([t for t in turns if t["role"] == "user"]),
        "turns": turns,
        "source": "llama3.1-8b-benign",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=str, default=f"{OUTPUT_DIR}/benign_llama.jsonl")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    print(f"=== Benign Llama Generation ===")
    print(f"Count: {args.count}, GPU: {device}, Output: {args.output}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = find_model_path()
    print(f"Loading model: {model_path} -> {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device
    )
    model.eval()
    print("Model loaded.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = []
    start_time = time.time()

    for i in range(args.count):
        scenario_info = SCENARIOS[i % len(SCENARIOS)]
        conv_id = f"benign_llama_{i:04d}"
        register_template = REGISTER_CASUAL if i % 2 == 0 else REGISTER_FORMAL
        register_name = "casual" if i % 2 == 0 else "formal"

        t0 = time.time()
        conv = generate_single_conversation(
            scenario_info, conv_id, register_template, register_name,
            model, tokenizer, device
        )
        elapsed = time.time() - t0
        results.append(conv)

        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        total_elapsed = time.time() - start_time
        avg = total_elapsed / (i + 1)
        eta = avg * (args.count - i - 1)
        print(f"[{i+1}/{args.count}] {conv_id} | reg={register_name} | turns={conv['num_turns']} | {elapsed:.1f}s | ETA {eta/60:.1f}m")

    total_time = (time.time() - start_time) / 60
    print(f"\nGenerated {len(results)} benign conversations in {total_time:.1f} min")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

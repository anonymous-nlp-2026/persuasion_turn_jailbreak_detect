"""Quick test: generate 5 conversations to verify <think> fix."""
import sys, os
sys.path.insert(0, '.')
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "~/.cache/huggingface"

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from scripts.generate_crescendo_v2 import (
    generate_conversations_plan, generate_single_conversation, 
    load_models, build_strategy_sequence
)
import json, time, re

print("Loading models...")
models = load_models()
print("Models loaded. Generating 5 test conversations...\n")

plan = generate_conversations_plan()[:5]

for i, plan_item in enumerate(plan):
    t0 = time.time()
    conv = generate_single_conversation(plan_item, models)
    elapsed = time.time() - t0
    
    user_turns = [t for t in conv["turns"] if t["role"] == "user"]
    has_think = any("<think>" in t["content"] for t in user_turns)
    has_empty = any(len(t["content"].strip()) == 0 for t in user_turns)
    
    print(f"\n{'='*60}")
    print(f"Conv {i+1}: {conv['conv_id']} | {conv['playbook_name']} | {elapsed:.1f}s")
    print(f"  Turns: {conv['num_turns']} | <think> leak: {has_think} | Empty: {has_empty}")
    
    for j, t in enumerate(user_turns):
        print(f"  [USER t{j}] s={t['intended_strategy']} ({t['strategy_name']}): {t['content'][:150]}")
    
    if has_think or has_empty:
        print("  *** PROBLEM DETECTED ***")

print("\n\nTest complete.")

# DisPeD: A Controlled Study of Encoder Adaptation for Multi-Turn Jailbreak Detection

This repository contains the code and data for the paper *"DisPeD: A Controlled Study of Encoder Adaptation for Multi-Turn Jailbreak Detection"*.

## Overview

Multi-turn jailbreak attacks distribute adversarial intent across individually benign conversation turns, evading single-turn safety classifiers. **DisPeD** (Disentangled Persuasion Detection) is a controlled framework that isolates how domain exposure, supervision type, and label semantics each contribute to out-of-distribution (OOD) robustness. The two-stage pipeline consists of:

1. **Stage 1 — Turn-Level Encoder Adaptation**: Seven DeBERTa-v3-base encoder variants share the same training data but differ solely in their pretraining objective (9-class persuasion classification, binary, scrambled labels, MLM, topic classification, Wikipedia MLM, or vanilla), each isolating a single factor.
2. **Stage 2 — Conversation-Level Classification**: A frozen BiGRU aggregates [CLS] embeddings from the frozen Stage-1 encoder for binary jailbreak detection.

Key findings:
- Classification-based domain adaptation (DAPT) is the only consistently effective mechanism, recovering OOD detection where vanilla encoders collapse (F1 0.97--1.0 vs. 0.26--0.31).
- Scrambled labels match correct labels -- the classification objective's structural role, not label content, drives generalization.
- These advantages reverse on human-authored data, where unsupervised pretraining leads, revealing condition-specific rather than universal advantages.
- The 184M-parameter pipeline achieves 24x speedup over LLM-as-judge alternatives.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA-capable GPU (tested on A6000, L40)

```bash
pip install torch torchvision torchaudio
pip install transformers datasets accelerate
pip install scikit-learn scipy tqdm
```

## Project Structure

```
src/
├── models/
│   ├── deberta_multitask.py    # DeBERTa multi-task model (Stage 1)
│   ├── deberta_topic.py        # Topic classification variant
│   ├── gru_classifier.py       # BiGRU classifier (Stage 2)
│   └── baseline.py             # Baseline model variants
├── train_deberta.py            # Stage 1: 9-class DAPT training
├── train_deberta_binary.py     # Stage 1: Binary variant
├── train_deberta_mlm.py        # Stage 1: MLM continued pretraining
├── train_deberta_mlm_wiki.py   # Stage 1: Wikipedia MLM control
├── train_deberta_topic.py      # Stage 1: Topic classification control
├── train_classifier.py         # Stage 2: BiGRU classifier training
├── evaluate.py                 # Evaluation pipeline
├── scramble_labels.py          # Label scrambling for scrambled variant
└── ...
scripts/                        # Experiment and analysis scripts
results/                        # Experiment result JSONs
docs/paper/                     # Paper source (LaTeX)
```

## Training

### Stage 1: Encoder Adaptation (Seven Variants)

**9-class DAPT** (primary variant):
```bash
python src/train_deberta.py \
    --train_data data/train.jsonl \
    --val_data data/val.jsonl \
    --model_name microsoft/deberta-v3-base \
    --output_dir checkpoints/stage1_9class \
    --epochs 5 --batch_size 16 --lr 2e-5
```

**MLM-only** (unsupervised control):
```bash
python src/train_deberta_mlm.py \
    --train_data data/train.jsonl \
    --model_name microsoft/deberta-v3-base \
    --output_dir checkpoints/stage1_mlm \
    --epochs 5 --batch_size 16
```

**Wikipedia MLM** (domain specificity control):
```bash
python src/train_deberta_mlm_wiki.py \
    --model_name microsoft/deberta-v3-base \
    --output_dir checkpoints/stage1_wiki_mlm \
    --epochs 5 --batch_size 16
```

**Topic classification** (task alignment control):
```bash
python src/train_deberta_topic.py \
    --train_data data/train.jsonl \
    --model_name microsoft/deberta-v3-base \
    --output_dir checkpoints/stage1_topic \
    --epochs 5 --batch_size 16
```

### Stage 2: Conversation-Level Classification

```bash
python src/train_classifier.py \
    --train_data data/train.jsonl \
    --val_data data/val.jsonl \
    --deberta_path checkpoints/stage1_9class/best \
    --output_dir checkpoints/stage2 \
    --epochs 30 --batch_size 8
```

### Evaluation

```bash
python src/evaluate.py \
    --test_data data/test.jsonl \
    --deberta_path checkpoints/stage1_9class/best \
    --gru_path checkpoints/stage2/best.pt \
    --output results/eval_results.json
```

## Data

Training data consists of synthetic multi-turn conversations with per-turn persuasion strategy annotations. Four attack families are used: Crescendo, Foot-in-the-Door (FITD), Deceptive Delight, and ActorAttack, generated via Qwen and Llama models. Strategy annotations are produced by Qwen3-8B.

The expected data format is JSONL with each line containing a conversation object with per-turn `strategy` labels and a conversation-level `label` field.

## License

CC BY-NC 4.0

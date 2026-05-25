# What Makes Multi-Turn Jailbreaks Detectable? Domain Exposure Bridges the Cross-Attack Gap

This repository contains the code and experimental results for the paper *"What Makes Multi-Turn Jailbreaks Detectable? Domain Exposure Bridges the Cross-Attack Gap"*.

## Overview

**DisPeD** is a controlled framework that isolates how domain exposure, supervision type, and label semantics each contribute to out-of-distribution (OOD) robustness in multi-turn jailbreak detection. The two-stage pipeline consists of:

1. **Stage 1 — DeBERTa Multi-Task Fine-Tuning**: Fine-tunes DeBERTa-v3-base on per-turn persuasion strategy classification (9-class) with an auxiliary binary jailbreak intent head.
2. **Stage 2 — GRU Sequence Classifier**: Trains a BiGRU classifier on conversation-level embeddings from the frozen Stage-1 encoder for binary jailbreak detection.

The resulting 184M-parameter pipeline achieves 24x speedup over LLM-as-judge alternatives.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA-capable GPU (tested on A6000, L40)

Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install transformers datasets accelerate einops
pip install scikit-learn scipy tqdm
```

## Project Structure

```
src/
├── data/
│   ├── dataset.py          # TurnDataset & ConversationDataset
│   └── collator.py         # Data collators for batching
├── models/
│   ├── deberta_multitask.py  # DeBERTa multi-task model (Stage 1)
│   ├── gru_classifier.py     # BiGRU classifier (Stage 2)
│   └── baseline.py           # Baseline model variants
├── train_deberta.py         # Stage 1: DeBERTa multi-task training
├── train_classifier.py      # Stage 2: GRU classifier training (9-class)
├── train_classifier_binary.py # Stage 2: GRU classifier training (binary)
├── evaluate.py              # Evaluation (F1, early detection, FPR@95TPR)
├── train_deberta_mlm.py     # MLM continued pretraining control
└── ...
scripts/                     # Experiment scripts (data generation, ablations, analysis)
results/                     # Experiment result JSONs
docs/paper/                  # Paper source (LaTeX)
```

## Data Preparation

The training data consists of synthetic multi-turn conversations with per-turn persuasion strategy annotations. To generate the data:

1. **Jailbreak conversations**: Generated using Crescendo-style multi-turn attacks with persuasion strategy labels.
   ```bash
   python scripts/generate_crescendo_v2.py
   ```

2. **Benign conversations**: Topic-anchored benign multi-turn dialogues.
   ```bash
   python src/generate_topic_anchored_benign_v3.py
   ```

3. **Merge and split**: Combine and create train/val/test splits.
   ```bash
   python src/merge_and_split.py
   ```

The expected data format is JSONL with each line containing a conversation object with per-turn `strategy` labels and a conversation-level `label` field.

## Training

### Stage 1: DeBERTa Multi-Task Fine-Tuning (DAPT)

```bash
python src/train_deberta.py \
    --train_data data/train.jsonl \
    --val_data data/val.jsonl \
    --model_name microsoft/deberta-v3-base \
    --output_dir checkpoints/stage1 \
    --epochs 5 \
    --batch_size 16 \
    --lr 2e-5
```

### Stage 2: GRU Sequence Classifier

```bash
# Treatment (persuasion-adapted encoder)
python src/train_classifier.py \
    --train_data data/train.jsonl \
    --val_data data/val.jsonl \
    --mode treatment \
    --deberta_path checkpoints/stage1/best \
    --output_dir checkpoints/stage2_treatment \
    --epochs 30 \
    --batch_size 8

# Baseline (vanilla DeBERTa encoder)
python src/train_classifier.py \
    --train_data data/train.jsonl \
    --val_data data/val.jsonl \
    --mode baseline \
    --output_dir checkpoints/stage2_baseline \
    --epochs 30 \
    --batch_size 8
```

### Evaluation

```bash
python src/evaluate.py \
    --test_data data/test.jsonl \
    --deberta_path checkpoints/stage1/best \
    --gru_path checkpoints/stage2_treatment/best.pt \
    --output results/eval_results.json
```

## Key Experiments

Experiment scripts are in `scripts/`. Notable ones:

| Script | Description |
|--------|-------------|
| `scripts/exp3_probing.py` | Frozen probing analysis |
| `scripts/exp28_data_efficiency.py` | Data efficiency ablation |
| `scripts/exp34_representation_tsne.py` | t-SNE visualization |
| `scripts/exp40_llama_guard_baseline.py` | Llama Guard baseline |
| `scripts/exp46_inference_efficiency.py` | Inference efficiency benchmark |
| `scripts/exp53_llm_judge_optimized.py` | LLM-as-judge baseline |
| `scripts/exp57_svd_iid_only.py` | SVD subspace analysis |

## License

This project is for research purposes.

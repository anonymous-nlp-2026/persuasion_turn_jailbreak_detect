"""
Scrambled-label ablation for DisPeD.
Randomly permutes persuasion_strategy labels within jailbreak conversations,
preserving the label distribution but breaking turn-strategy correspondence.

Usage:
    python src/scramble_labels.py --input data/final_v2/train.jsonl --output data/final_v2/train_scrambled.jsonl --seed 42
"""
import json, argparse, random


def scramble_strategies(conversations, seed=42):
    rng = random.Random(seed)
    result = []

    for conv in conversations:
        conv_copy = json.loads(json.dumps(conv))

        if conv_copy.get('label') == 'jailbreak':
            strategies = []
            for turn in conv_copy['turns']:
                if turn.get('role') == 'user':
                    s = turn.get('persuasion_label', turn.get('persuasion_strategy', 0))
                    strategies.append(s)

            rng.shuffle(strategies)

            idx = 0
            for turn in conv_copy['turns']:
                if turn.get('role') == 'user':
                    if 'persuasion_label' in turn:
                        turn['persuasion_label'] = strategies[idx]
                    if 'persuasion_strategy' in turn:
                        turn['persuasion_strategy'] = strategies[idx]
                    idx += 1

        result.append(conv_copy)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    with open(args.input) as f:
        data = [json.loads(l) for l in f]

    scrambled = scramble_strategies(data, args.seed)

    with open(args.output, 'w') as f:
        for item in scrambled:
            f.write(json.dumps(item) + '\n')

    jb = [c for c in scrambled if c.get('label') == 'jailbreak']
    print(f"Processed {len(scrambled)} conversations")
    print(f"  Jailbreak (scrambled): {len(jb)}")
    print(f"  Benign (unchanged): {len(scrambled) - len(jb)}")

"""
Experiment script for analyzing grokking via mechanistic interpretability.

Run this after training to analyze your model's Fourier structure.
"""

import torch
import os
import argparse
from model import ModAddModel
from data import make_mod_add_loaders
from analysis import (
    analyze_embedding_fourier,
    analyze_logit_fourier,
    compute_grokking_progress,
    plot_attention_patterns,
    analyze_checkpoints,
    plot_grokking_progress,
)
from dataclasses import dataclass


@dataclass
class ModelConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    h_q: int
    h_k: int
    d_model: int
    d_head: int


def load_model_from_checkpoint(checkpoint_path: str, p: int) -> ModAddModel:
    """Load a model from a checkpoint file."""
    model_config = ModelConfig(
        block_size=4,
        vocab_size=p + 1,
        n_layer=1,
        h_q=4,
        h_k=4,
        d_model=128,
        d_head=32,
    )

    model = ModAddModel(model_config)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    return model, model_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, required=True, help='Directory containing checkpoints')
    parser.add_argument('--p', type=int, default=113, help='Prime modulus')
    parser.add_argument('--checkpoint', type=str, default='epoch_final.pt', help='Which checkpoint to analyze')
    args = parser.parse_args()

    checkpoint_path = os.path.join(args.run_dir, 'checkpoints', args.checkpoint)
    checkpoint_dir = os.path.join(args.run_dir, 'checkpoints')

    print(f"Loading model from {checkpoint_path}")
    model, model_config = load_model_from_checkpoint(checkpoint_path, args.p)

    print(f"\n{'='*60}")
    print("MECHANISTIC INTERPRETABILITY ANALYSIS")
    print(f"{'='*60}\n")

    # 1. Analyze embedding Fourier structure
    print("1. Embedding Fourier Analysis")
    print("-" * 40)
    embed_result = analyze_embedding_fourier(model, args.p, plot=True)
    print(f"Top 5 Fourier frequencies: {embed_result['top_frequencies'].tolist()}")
    print(f"Power at top frequency: {embed_result['power_fraction'][embed_result['top_frequencies'][0]]:.4f}")
    print()

    # 2. Analyze unembedding Fourier structure
    print("2. Unembedding Fourier Analysis")
    print("-" * 40)
    unembed_result = analyze_logit_fourier(model, args.p, plot=True)
    print(f"Top 5 Fourier frequencies: {unembed_result['top_frequencies'].tolist()}")
    print()

    # 3. Compute grokking progress
    print("3. Grokking Progress Measures")
    print("-" * 40)
    progress = compute_grokking_progress(model, args.p)
    print(f"Key frequencies: {progress['key_frequencies']}")
    print(f"Embedding key power: {progress['embed_key_power']:.4f}")
    print(f"Unembedding key power: {progress['unembed_key_power']:.4f}")
    print(f"Total key power: {progress['total_key_power']:.4f}")
    print()

    # 4. Visualize attention patterns
    print("4. Attention Pattern Analysis")
    print("-" * 40)
    _, val_loader = make_mod_add_loaders(args.p, train_frac=0.3, batch_size=1)
    sample_x, sample_y = next(iter(val_loader))
    sample_x = sample_x[:1]  # Just one example
    print(f"Example input: {sample_x[0].tolist()} (represents {sample_x[0,0].item()} + {sample_x[0,1].item()} = ?)")
    plot_attention_patterns(model, sample_x)
    print()

    # 5. Analyze training progression (if checkpoints exist)
    print("5. Training Progression Analysis")
    print("-" * 40)
    if os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 1:
        print("Analyzing checkpoints...")
        progression = analyze_checkpoints(
            checkpoint_dir, ModAddModel, model_config, args.p
        )
        plot_grokking_progress(progression)

        # Find grokking point
        val_accs = progression['val_accs']
        for i, acc in enumerate(val_accs):
            if acc > 0.95:
                print(f"Model reached >95% validation accuracy at epoch {progression['epochs'][i]}")
                break
    else:
        print("Not enough checkpoints for progression analysis")

    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}\n")

    print("""
The model learns modular addition using Fourier components:

1. EMBEDDING: Numbers are embedded as rotations in 2D subspaces.
   Each Fourier frequency k corresponds to a rotation by 2*pi*k/p.

2. COMPUTATION: The attention/MLP layers compose rotations using
   trigonometric identities: cos(a+b) = cos(a)cos(b) - sin(a)sin(b)

3. UNEMBEDDING: The result is read off by projecting back through
   the inverse Fourier basis.

If your model has:
- Strong power at a few specific Fourier frequencies -> It learned the algorithm!
- Power spread across many frequencies -> Still memorizing
- Top frequencies match between embed and unembed -> Clean algorithm
""")


if __name__ == "__main__":
    main()

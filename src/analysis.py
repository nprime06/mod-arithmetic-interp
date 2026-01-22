"""
Mechanistic Interpretability Analysis Tools for Grokking

Based on Neel Nanda's work: "Progress measures for grokking via mechanistic interpretability"
https://arxiv.org/abs/2301.05217

The key insight: the model learns to compute (a + b) mod p using Fourier components.
It embeds numbers as rotations in 2D and composes them using trig identities.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import os


def fourier_basis(p: int, device='cpu') -> torch.Tensor:
    """
    Create the Fourier basis for mod p arithmetic.

    Returns a (p, p) matrix where each row k contains:
    - k=0: constant term (1/sqrt(p))
    - k=1 to p//2: cos(2*pi*k*n/p) terms
    - k=p//2+1 to p-1: sin(2*pi*k*n/p) terms

    This is the discrete Fourier transform basis.
    """
    basis = torch.zeros(p, p, device=device)
    basis[0] = 1 / np.sqrt(p)  # Constant term

    for k in range(1, p // 2 + 1):
        # Cosine terms
        basis[2*k - 1] = torch.cos(2 * np.pi * k * torch.arange(p, device=device) / p) * np.sqrt(2/p)
        # Sine terms (if k < p//2)
        if 2*k < p:
            basis[2*k] = torch.sin(2 * np.pi * k * torch.arange(p, device=device) / p) * np.sqrt(2/p)

    return basis


def compute_fourier_components(embedding: torch.Tensor, p: int) -> torch.Tensor:
    """
    Project embedding matrix onto Fourier basis.

    Args:
        embedding: (p, d_model) embedding matrix (first p rows, excluding = token)
        p: prime modulus

    Returns:
        (p, d_model) matrix of Fourier coefficients
    """
    basis = fourier_basis(p, device=embedding.device)
    # Project: F @ E gives us the Fourier components
    return basis @ embedding


def compute_fourier_power(embedding: torch.Tensor, p: int) -> torch.Tensor:
    """
    Compute power spectrum of Fourier components for each frequency.

    Returns (p,) tensor where entry k is the total power at frequency k.
    """
    fourier = compute_fourier_components(embedding, p)
    # Sum squared magnitude across embedding dimensions
    power = (fourier ** 2).sum(dim=1)
    return power


def analyze_embedding_fourier(model, p: int, plot: bool = True) -> dict:
    """
    Analyze the Fourier structure of the token embedding.

    Key finding from Nanda: After grokking, the embedding should have
    strong power at specific Fourier frequencies that enable the
    rotation-based computation.
    """
    with torch.no_grad():
        embedding = model.model.wte.weight[:p].detach()  # First p tokens (numbers 0 to p-1)
        power = compute_fourier_power(embedding, p)

        # Normalize to get fraction of power at each frequency
        power_frac = power / power.sum()

        # Find top frequencies
        top_freqs = torch.argsort(power, descending=True)[:5]

    if plot:
        plt.figure(figsize=(10, 4))
        plt.bar(range(p), power_frac.cpu().numpy())
        plt.xlabel('Fourier frequency')
        plt.ylabel('Fraction of power')
        plt.title('Fourier power spectrum of embeddings')
        plt.show()

    return {
        'power': power,
        'power_fraction': power_frac,
        'top_frequencies': top_freqs,
    }


def analyze_logit_fourier(model, p: int, plot: bool = True) -> dict:
    """
    Analyze Fourier structure of the unembedding (lm_head) matrix.

    For the rotation algorithm to work, the unembedding should also
    have power concentrated at the same frequencies as the embedding.
    """
    with torch.no_grad():
        unembed = model.lm_head.weight[:p].detach()  # (p, d_model)
        power = compute_fourier_power(unembed, p)
        power_frac = power / power.sum()
        top_freqs = torch.argsort(power, descending=True)[:5]

    if plot:
        plt.figure(figsize=(10, 4))
        plt.bar(range(p), power_frac.cpu().numpy())
        plt.xlabel('Fourier frequency')
        plt.ylabel('Fraction of power')
        plt.title('Fourier power spectrum of unembedding')
        plt.show()

    return {
        'power': power,
        'power_fraction': power_frac,
        'top_frequencies': top_freqs,
    }


def compute_restricted_loss(model, data_loader, p: int, excluded_freqs: list) -> float:
    """
    Compute loss after ablating certain Fourier frequencies from embeddings.

    This tests whether specific frequencies are necessary for the computation.
    If removing a frequency increases loss, that frequency is important.
    """
    model.eval()
    with torch.no_grad():
        # Get original embedding
        original_embed = model.model.wte.weight.clone()

        # Create ablation mask in Fourier space
        basis = fourier_basis(p, device=original_embed.device)

        # Project to Fourier, zero out excluded frequencies, project back
        embed_numbers = original_embed[:p]  # Numbers 0 to p-1
        fourier = basis @ embed_numbers

        for freq in excluded_freqs:
            fourier[freq] = 0

        # Project back
        ablated_embed = basis.T @ fourier

        # Replace embedding temporarily
        model.model.wte.weight.data[:p] = ablated_embed

        # Compute loss
        total_loss = 0
        total_samples = 0
        for x, y in data_loader:
            output = model(x)[:, -1, :]
            loss = F.cross_entropy(output, y, reduction='sum')
            total_loss += loss.item()
            total_samples += y.size(0)

        # Restore original embedding
        model.model.wte.weight.data = original_embed

    return total_loss / total_samples


def compute_grokking_progress(model, p: int, key_freqs: Optional[list] = None) -> dict:
    """
    Compute progress measures that track grokking.

    Key insight from Nanda: The Fourier components grow gradually during
    training, even though test accuracy jumps suddenly. This provides
    a continuous "progress measure" for grokking.

    Args:
        model: trained model
        p: prime modulus
        key_freqs: list of key Fourier frequencies to track. If None, uses all.

    Returns dict with:
        - embed_power: power at each Fourier frequency in embedding
        - unembed_power: power at each Fourier frequency in unembedding
        - total_key_power: sum of power at key frequencies (progress measure)
    """
    embed_result = analyze_embedding_fourier(model, p, plot=False)
    unembed_result = analyze_logit_fourier(model, p, plot=False)

    if key_freqs is None:
        # Use top 5 frequencies from embedding as key frequencies
        key_freqs = embed_result['top_frequencies'][:5].tolist()

    embed_key_power = embed_result['power'][key_freqs].sum().item()
    unembed_key_power = unembed_result['power'][key_freqs].sum().item()

    return {
        'embed_power': embed_result['power'],
        'unembed_power': unembed_result['power'],
        'key_frequencies': key_freqs,
        'embed_key_power': embed_key_power,
        'unembed_key_power': unembed_key_power,
        'total_key_power': embed_key_power + unembed_key_power,
    }


def plot_attention_patterns(model, x: torch.Tensor, head_idx: int = 0):
    """
    Visualize attention patterns for a given input.

    After grokking, attention heads should show clear patterns:
    - Some heads attend uniformly
    - Some heads attend specifically to position of 'a' or 'b'
    """
    model.eval()
    with torch.no_grad():
        _ = model(x)
        # Get attention pattern from first (and only) transformer block
        attn = model.model.h[0].gqa.attn_pattern  # Stored during forward pass

    # attn shape: (B, h_q/h_k, h_k, T, T)
    attn_2d = attn[0].reshape(-1, attn.shape[-2], attn.shape[-1])  # (num_heads, T, T)

    fig, axes = plt.subplots(1, min(4, attn_2d.shape[0]), figsize=(12, 3))
    if attn_2d.shape[0] == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        if i < attn_2d.shape[0]:
            im = ax.imshow(attn_2d[i].cpu().numpy(), cmap='Blues')
            ax.set_title(f'Head {i}')
            ax.set_xlabel('Key position')
            ax.set_ylabel('Query position')
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


def analyze_checkpoints(checkpoint_dir: str, model_class, model_config, p: int,
                        data_loader=None) -> dict:
    """
    Analyze Fourier progress measures across training checkpoints.

    This reveals how the algorithm forms gradually during training,
    even though test accuracy jumps suddenly (grokking).
    """
    import glob

    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt")))

    epochs = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    key_powers = []

    for ckpt_path in checkpoint_files:
        ckpt = torch.load(ckpt_path, map_location='cpu')

        # Create fresh model and load weights
        model = model_class(model_config)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        epochs.append(ckpt['epoch'])
        train_losses.append(ckpt.get('train_loss', 0))
        val_losses.append(ckpt.get('val_loss', 0))
        train_accs.append(ckpt.get('train_acc', 0))
        val_accs.append(ckpt.get('val_acc', 0))

        # Compute Fourier progress
        progress = compute_grokking_progress(model, p)
        key_powers.append(progress['total_key_power'])

    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'key_powers': key_powers,
    }


def plot_grokking_progress(analysis_result: dict):
    """
    Plot training curves alongside Fourier progress measure.

    Key insight: The Fourier power increases gradually, while
    validation accuracy jumps suddenly. This shows grokking is
    actually a continuous process internally.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = analysis_result['epochs']

    # Loss curves
    axes[0].plot(epochs, analysis_result['train_losses'], label='Train')
    axes[0].plot(epochs, analysis_result['val_losses'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].set_yscale('log')

    # Accuracy curves
    axes[1].plot(epochs, analysis_result['train_accs'], label='Train')
    axes[1].plot(epochs, analysis_result['val_accs'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()

    # Fourier progress measure
    axes[2].plot(epochs, analysis_result['key_powers'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Key Fourier Power')
    axes[2].set_title('Fourier Progress Measure')

    plt.tight_layout()
    plt.show()


def plot_rotation_circles(model, p: int, fixed_x: int = 0, top_k_freqs: int = 4,
                          save_path: Optional[str] = None):
    """
    Visualize how neuron activations trace circles as y varies (with x fixed).

    This demonstrates the Fourier rotation structure: the model encodes numbers
    as rotations in 2D subspaces, so as y varies from 0 to p-1, activations
    should trace circles for neurons responding to the same frequency.

    Args:
        model: trained ModAddModel
        p: prime modulus
        fixed_x: fixed value for first operand
        top_k_freqs: number of top Fourier frequencies to visualize
        save_path: optional path to save the figure
    """
    model.eval()

    # Create inputs: (fixed_x, y, =) for y in 0..p-1
    inputs = torch.tensor([[fixed_x, y, p] for y in range(p)])

    with torch.no_grad():
        # Get embeddings for the y position (index 1)
        y_values = inputs[:, 1]
        embeddings = model.model.wte(y_values)  # (p, d_model)

        # Run forward pass to get MLP activations
        _ = model(inputs)
        mlp_pre = model.model.h[0].mlp_pre[:, 1, :]  # (p, d_model*4) at position 1

    # Project embeddings onto Fourier basis
    basis = fourier_basis(p, device=embeddings.device)
    fourier_coeffs = basis @ embeddings  # (p, d_model)

    # Get top frequencies from embedding power
    embed_power = (fourier_coeffs ** 2).sum(dim=1)
    top_freqs = torch.argsort(embed_power, descending=True)[:top_k_freqs].tolist()

    # Create figure: 2 rows (embedding, MLP) x top_k_freqs columns
    fig, axes = plt.subplots(2, top_k_freqs, figsize=(4 * top_k_freqs, 8))

    # Color by y value (use cyclic colormap)
    colors = plt.cm.hsv(np.linspace(0, 1, p))

    # Row 1: Embedding Fourier components
    for i, freq in enumerate(top_freqs):
        ax = axes[0, i]

        # For frequency k, the Fourier basis has:
        # - cos component at index 2k-1 (for k >= 1)
        # - sin component at index 2k (for k >= 1, if 2k < p)
        if freq == 0:
            # DC component - just plot first two dimensions
            x_comp = fourier_coeffs[0, :].cpu().numpy()
            y_comp = fourier_coeffs[1, :].cpu().numpy() if fourier_coeffs.shape[0] > 1 else np.zeros(p)
            ax.set_title(f'Freq 0 (DC)', fontsize=12)
        else:
            # Get the actual frequency number from the basis index
            # Basis is organized: [DC, cos1, sin1, cos2, sin2, ...]
            # So freq index k corresponds to frequency ceil(k/2)
            actual_freq = (freq + 1) // 2

            cos_idx = 2 * actual_freq - 1
            sin_idx = 2 * actual_freq

            if sin_idx < p:
                # Sum the power across embedding dimensions for this frequency pair
                cos_vals = fourier_coeffs[cos_idx, :].cpu().numpy()
                sin_vals = fourier_coeffs[sin_idx, :].cpu().numpy()

                # Use PCA on the 2D Fourier subspace to find the circle
                # Or just use the first two dimensions with highest variance
                x_comp = cos_vals
                y_comp = sin_vals
            else:
                x_comp = fourier_coeffs[freq, :].cpu().numpy()
                y_comp = np.zeros_like(x_comp)

            ax.set_title(f'Freq {actual_freq} (basis idx {freq})', fontsize=12)

        # Actually, let's simplify: plot the Fourier coefficient magnitude
        # at this frequency index for each y value
        # The coefficient for input y at frequency k should trace a circle

        # Get cos and sin basis vectors
        cos_basis = torch.cos(2 * np.pi * ((freq + 1) // 2) * torch.arange(p, device=embeddings.device).float() / p)
        sin_basis = torch.sin(2 * np.pi * ((freq + 1) // 2) * torch.arange(p, device=embeddings.device).float() / p)

        # Project each y's embedding onto cos and sin
        cos_proj = (embeddings * cos_basis.unsqueeze(1)).sum(dim=0)  # (d_model,)
        sin_proj = (embeddings * sin_basis.unsqueeze(1)).sum(dim=0)  # (d_model,)

        # For each y, compute its position in the cos-sin plane
        # The embedding of y has some component along cos and sin directions
        actual_freq_k = (freq + 1) // 2
        x_circle = torch.cos(2 * np.pi * actual_freq_k * torch.arange(p).float() / p).numpy()
        y_circle = torch.sin(2 * np.pi * actual_freq_k * torch.arange(p).float() / p).numpy()

        # Scale by the power at this frequency
        scale = np.sqrt(embed_power[freq].item())
        x_circle *= scale
        y_circle *= scale

        for j in range(p):
            ax.scatter(x_circle[j], y_circle[j], c=[colors[j]], s=30, alpha=0.8)

        ax.set_xlabel('cos component')
        ax.set_ylabel('sin component')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    axes[0, 0].set_ylabel('Embedding\nFourier Space', fontsize=12, fontweight='bold')

    # Row 2: MLP neuron pairs
    # Find neurons that correlate with cos/sin of top frequencies
    for i, freq in enumerate(top_freqs):
        ax = axes[1, i]
        actual_freq_k = (freq + 1) // 2

        # Ideal cos and sin for this frequency
        cos_ideal = torch.cos(2 * np.pi * actual_freq_k * torch.arange(p).float() / p)
        sin_ideal = torch.sin(2 * np.pi * actual_freq_k * torch.arange(p).float() / p)

        # Find MLP neurons most correlated with cos and sin
        mlp_pre_np = mlp_pre.cpu().numpy()
        cos_corrs = np.array([np.corrcoef(mlp_pre_np[:, n], cos_ideal.numpy())[0, 1]
                              for n in range(mlp_pre_np.shape[1])])
        sin_corrs = np.array([np.corrcoef(mlp_pre_np[:, n], sin_ideal.numpy())[0, 1]
                              for n in range(mlp_pre_np.shape[1])])

        # Handle NaN correlations
        cos_corrs = np.nan_to_num(cos_corrs, 0)
        sin_corrs = np.nan_to_num(sin_corrs, 0)

        best_cos_neuron = np.argmax(np.abs(cos_corrs))
        best_sin_neuron = np.argmax(np.abs(sin_corrs))

        # Plot the two neurons against each other
        x_vals = mlp_pre_np[:, best_cos_neuron]
        y_vals = mlp_pre_np[:, best_sin_neuron]

        for j in range(p):
            ax.scatter(x_vals[j], y_vals[j], c=[colors[j]], s=30, alpha=0.8)

        ax.set_xlabel(f'Neuron {best_cos_neuron} (cos corr: {cos_corrs[best_cos_neuron]:.2f})')
        ax.set_ylabel(f'Neuron {best_sin_neuron} (sin corr: {sin_corrs[best_sin_neuron]:.2f})')
        ax.set_title(f'MLP neurons for freq {actual_freq_k}', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    axes[1, 0].set_ylabel('MLP Pre-ReLU\nNeuron Pairs', fontsize=12, fontweight='bold')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='hsv', norm=plt.Normalize(vmin=0, vmax=p-1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label('y value (color = rotation angle)', fontsize=11)

    plt.suptitle(f'Rotation Circles: x={fixed_x} fixed, y varies 0→{p-1}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Fourier Analysis Tools for Grokking")
    print("====================================")
    print()
    print("After training, use these functions to analyze your model:")
    print()
    print("1. analyze_embedding_fourier(model, p) - See Fourier structure of embeddings")
    print("2. analyze_logit_fourier(model, p) - See Fourier structure of unembedding")
    print("3. compute_grokking_progress(model, p) - Get progress measures")
    print("4. plot_attention_patterns(model, x) - Visualize attention")
    print("5. analyze_checkpoints(...) - Track progress across training")
    print("6. plot_rotation_circles(model, p) - Visualize rotation structure")

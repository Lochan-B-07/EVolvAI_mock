"""
generative_core/train.py
========================
Training loop for the GenerativeCounterfactualVAE.

Trains on the dataset returned by `get_dataloader()` using the Adam optimiser
with gradient clipping and saves a model state-dict checkpoint on completion.

Typical usage
-------------
    # From the CLI (recommended):
    python run.py train

    # Programmatically (e.g. from run.py 'all' mode):
    from generative_core.train import train
    model, device = train(epochs=50, save=True)
"""

import os

import torch
import torch.optim as optim

from . import config
from .models import GenerativeCounterfactualVAE, vae_loss_function
from .data_loader import get_dataloader


def train(epochs: int = config.EPOCHS, save: bool = True):
    """Train the GCD-VAE from scratch and optionally save a checkpoint.

    The training condition is the BASELINE_CONDITION vector (normal weekday
    with no weather anomaly).  The model learns to reconstruct historical
    demand from this latent representation, which gives it the posterior
    necessary to generate counterfactuals at inference time.

    Loss:  MSE reconstruction (mean-reduced) + β × KL divergence.
    Optimiser: Adam with initial LR = config.LEARNING_RATE.
    Gradient clipping at L2 norm config.GRAD_CLIP_NORM prevents explosion
    which is common on early epochs before the decoder stabilises.

    Args:
        epochs (int): Number of passes over the full dataset.
        save (bool):  Whether to write the state-dict to config.MODEL_SAVE_PATH.

    Returns:
        Tuple (model, device):
            model  – trained GenerativeCounterfactualVAE (in eval-ready state).
            device – torch.device the model lives on (for chaining with generate).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    model = GenerativeCounterfactualVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loader = get_dataloader()

    # Pre-build the condition tensor once; expand() creates a view per batch
    # without allocating new memory on every iteration.
    baseline_cond = torch.tensor(
        config.BASELINE_CONDITION, dtype=torch.float32, device=device,
    )

    from .physics_loss import LinDistFlowLoss
    physics_engine = LinDistFlowLoss(device)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        # Linear KLD warmup: β ramps 0.0 → 0.1 over first 50 epochs, then holds.
        # Physics lambdas (in config) are set to 0.0 during isolation training.
        current_kld = min(0.1, (epoch - 1) / 50.0)
            # DataLoader yields [B, seq_len, features]; Conv1d expects
            # [B, features, seq_len] – permute here, not in the dataset.
            x = batch.permute(0, 2, 1).to(device)

            # Expand the shared condition vector to match the current batch size.
            cond = baseline_cond.unsqueeze(0).expand(x.size(0), -1)

            optimizer.zero_grad()
            recon, mu, logvar = model(x, cond)

            # Calculate Physics Loss using the reconstructed EV demand (first config.NUM_NODES features)
            # Permute to [B, T, nodes] so time and batch map nicely
            ev_demand = recon[:, :config.NUM_NODES, :].permute(0, 2, 1)
            
            # The physics engine takes the demand and computes standard violations
            pen_v, pen_therm, pen_xfmr = physics_engine(ev_demand)
            physics_loss = (config.LAMBDA_VOLT * pen_v +
                            config.LAMBDA_THERMAL * pen_therm +
                            config.LAMBDA_XFMR * pen_xfmr)

            loss = vae_loss_function(recon, x, mu, logvar, physics_loss)
            loss.backward()

            # Clip gradient norm to avoid spikes on early epochs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)

            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        # Divide by batches (not dataset length) to stay consistent with the
        # mean-reduced loss inside vae_loss_function.
        avg = epoch_loss / max(num_batches, 1)
        print(f"  Epoch {epoch:>3}/{epochs}  avg_loss={avg:.6f}")

    if save:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        print(f"[train] Checkpoint saved → {config.MODEL_SAVE_PATH}")

    return model, device

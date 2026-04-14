"""
generative_core/models.py
==========================
PyTorch model definitions for the GCD-VAE pipeline.

Classes
-------
CausalConv1d
    A causal (past-only) 1D convolution built by pre-padding the input so
    that position t can only see positions ≤ t.  This is mandatory for
    autoregressive demand modelling – using a standard Conv1d here would let
    the encoder peek at future hours when reconstructing past ones.

TCNBlock
    One residual TCN block: two stacked CausalConv1d layers + skip connection.
    The skip connection (downsample when channel widths differ) stabilises
    training and lets gradients flow around any saturated layers.

TemporalConvNet
    Stacks multiple TCNBlocks with exponentially increasing dilation
    (1, 2, 4, ...) so the receptive field grows without stacking many layers.
    With KERNEL_SIZE=2 and two blocks: effective receptive field = 1 + (2-1)*1
    + (2-1)*2 = 4 time steps.

GenerativeCounterfactualVAE
    The core model.  Encoder extracts a Gaussian posterior q(Z|X), the
    reparameterisation trick draws a differentiable sample Z, and the decoder
    reconstructs X̂ conditioned on Z and C (the intervention vector).

    The only thing distinguishing this from a standard VAE is the condition
    vector C that is concatenated to Z before decoding.  At generation time,
    changing C with a trained Z produces "what-if" demand profiles without
    retraining.

vae_loss_function
    β-VAE objective: MSE reconstruction + β × KL divergence.
    Both terms are normalised per-element (reduction="mean") so the loss
    magnitude is independent of batch size – critical for stable LR tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


class CausalConv1d(nn.Conv1d):
    """1D convolution that is strictly causal (no look-ahead).

    Works by padding (kernel_size - 1) * dilation zeros on the left of the
    time axis before the convolution, then slicing off the right-edge artefact.

    Args:
        in_channels (int): Input feature channels.
        out_channels (int): Output feature channels.
        kernel_size (int): Convolution kernel width.
        stride (int): Convolution stride (almost always 1 in TCNs).
        dilation (int): Dilation factor – doubles for each successive block.
        groups (int): Grouped convolution multiplier.
        bias (bool): Whether to add a learnable bias term.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True):
        # Amount of left-padding needed to maintain causality.
        self._causal_padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=self._causal_padding,          # symmetric padding by default
            dilation=dilation, groups=groups, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        # Remove the right-hand overflow introduced by symmetric padding so
        # the output length equals the input length and causality is preserved.
        if self._causal_padding != 0:
            return out[:, :, :-self._causal_padding]
        return out


class TCNBlock(nn.Module):
    """Residual TCN block: Conv → ReLU → Dropout × 2 + skip connection.

    Args:
        n_inputs (int): Input channel count.
        n_outputs (int): Output channel count.
        kernel_size (int): Kernel width for each CausalConv1d.
        stride (int): Convolution stride (always 1 here).
        dilation (int): Dilation for this block (doubles with depth).
        dropout (float): Dropout probability after each activation.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 dropout=config.DROPOUT):
        super().__init__()
        self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size,
                                  stride=stride, dilation=dilation)
        self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size,
                                  stride=stride, dilation=dilation)
        self.net = nn.Sequential(
            self.conv1, nn.ReLU(), nn.Dropout(dropout),
            self.conv2, nn.ReLU(), nn.Dropout(dropout),
        )
        # 1×1 conv to project channels for the residual skip when widths differ.
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stack of TCNBlocks with exponentially growing dilation.

    The receptive field of a single block at depth i covers
    (kernel_size - 1) * 2^i time steps.  Stacking blocks gives the model access
    to the full 24-hour context with far fewer parameters than an LSTM.

    Args:
        num_inputs (int): Feature dimension of the input sequence.
        num_channels (list[int]): Output channels for each block.
        kernel_size (int): Kernel width (shared across blocks).
        dropout (float): Per-block dropout probability.
    """

    def __init__(self, num_inputs, num_channels,
                 kernel_size=config.KERNEL_SIZE, dropout=config.DROPOUT):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(
                TCNBlock(in_ch, out_ch, kernel_size,
                         stride=1, dilation=2 ** i, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GenerativeCounterfactualVAE(nn.Module):
    """Conditioned TCN-VAE that generates counterfactual EV demand profiles.

    The model learns to encode a [24, NUM_NODES+weather] demand tensor into a
    Gaussian posterior over Z ∈ ℝ^LATENT_DIM.  At inference, Z is sampled and
    decoded conditioned on a scenario vector C ∈ ℝ^COND_DIM to produce a
    new [24, NUM_NODES] demand forecast for that hypothetical scenario.

    Tensor shapes (using B = batch, F = NUM_FEATURES, T = SEQ_LEN):
        Input  x      : [B, F, T]   (features on the channel axis for Conv1d)
        Condition C   : [B, COND_DIM]
        Latent Z      : [B, LATENT_DIM]
        Output x̂     : [B, F, T]   (same shape as input)

    Args:
        num_features (int): Total input feature width (nodes + weather channels).
        seq_len (int): Time-series length (24 hours).
        latent_dim (int): VAE latent space dimensionality.
        cond_dim (int): Condition vector length.
    """

    def __init__(self,
                 num_features: int = config.NUM_FEATURES,
                 seq_len: int = config.SEQ_LEN,
                 latent_dim: int = config.LATENT_DIM,
                 cond_dim: int = config.COND_DIM):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        tcn_ch = config.TCN_CHANNELS                    # e.g. [32, 64]
        tcn_flat = seq_len * tcn_ch[-1]                 # flattened encoder output size

        # ── Encoder ──────────────────────────────────────────────────────
        self.encoder_tcn = TemporalConvNet(num_features, tcn_ch)
        self.fc_mu     = nn.Linear(tcn_flat, latent_dim)    # posterior mean
        self.fc_logvar = nn.Linear(tcn_flat, latent_dim)    # posterior log-variance

        # ── Decoder ──────────────────────────────────────────────────────
        # Accepts Z concatenated with C, projects back up to tcn_flat, then
        # reshapes for the decoder TCN.
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, config.DECODER_HIDDEN),
            nn.ReLU(),
            nn.Linear(config.DECODER_HIDDEN, tcn_flat),
            nn.ReLU(),
        )
        self.decoder_tcn = TemporalConvNet(tcn_ch[-1], [32, num_features])

    def encode(self, x: torch.Tensor):
        """Run the encoder TCN and return (μ, log σ²) of the posterior.

        Args:
            x: Input tensor [B, num_features, seq_len].

        Returns:
            Tuple of (mu, logvar), each [B, latent_dim].
        """
        h = self.encoder_tcn(x).flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample Z using the reparameterisation trick: Z = μ + ε·σ, ε ~ N(0,I).

        This keeps the sample on the computational graph so gradients can flow
        back through the sampling step to the encoder.

        Args:
            mu: Posterior mean [B, latent_dim].
            logvar: Posterior log-variance [B, latent_dim].

        Returns:
            Sampled latent vector Z [B, latent_dim].
        """
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Decode a latent sample Z conditioned on intervention vector C.

        Args:
            z: Latent vector [B, latent_dim].
            condition: Condition vector [B, cond_dim].

        Returns:
            Reconstructed demand tensor [B, num_features, seq_len].
        """
        h = self.decoder_fc(torch.cat([z, condition], dim=-1))
        h = h.view(h.size(0), config.TCN_CHANNELS[-1], self.seq_len)
        return self.decoder_tcn(h)

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        """Full VAE forward pass: encode → reparameterise → decode.

        Args:
            x: Input demand tensor [B, num_features, seq_len].
            condition: Baseline condition vector [B, cond_dim].

        Returns:
            Tuple (x_hat, mu, logvar):
                x_hat  – reconstructed tensor [B, num_features, seq_len]
                mu     – encoder mean [B, latent_dim]
                logvar – encoder log-variance [B, latent_dim]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, condition), mu, logvar


def vae_loss_function(recon_x: torch.Tensor, x: torch.Tensor,
                      mu: torch.Tensor, logvar: torch.Tensor,
                      physics_loss: torch.Tensor = torch.tensor(0.0),
                      current_kld_weight: float = 0.0) -> torch.Tensor:
    """β-VAE loss: MSE reconstruction + KL divergence + Physics Penalty.

    L = MSE(x̂, x) + current_kld_weight · KL[q(Z|X) ‖ N(0, I)] + Physics Penalty

    Args:
        recon_x: Reconstructed output [B, F, T].
        x:       Original input [B, F, T].
        mu:      Encoder mean [B, latent_dim].
        logvar:  Encoder log-variance [B, latent_dim].
        physics_loss: Aggregated LinDistFlow penalty scalar.
        current_kld_weight: Dynamic β passed in from the training loop.
                            Use 0.0 at epoch 1 and ramp up via annealing schedule.
                            Replaces the static config.KLD_WEIGHT so the caller
                            has full control over the annealing curve.

    Returns:
        Scalar loss tensor.
    """
    recon = F.mse_loss(recon_x, x, reduction="mean")
    kld   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + current_kld_weight * kld + physics_loss


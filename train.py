"""
train.py — EVolvAI standalone training script
==============================================
Run from repo root on any machine with Python + PyTorch:

    python3 train.py                       # default 500 epochs, synthetic data
    python3 train.py --epochs 1000         # full GPU run
    python3 train.py --help                # all options

Data priority (automatic):
    1. data/processed/train_data.parquet   ← real ACN data (if present)
    2. Synthetic fallback (Lochan generator via data_loader)

To build the real parquet first:
    python3 data_pipeline/preprocess.py --api-token YOUR_TOKEN
    python3 data_pipeline/preprocess.py --csv data/raw/acn_sessions.csv
    python3 data_pipeline/preprocess.py --synthetic --days 2000   # no creds
"""

import argparse
import datetime
import os
import sys
import time

import numpy as np

# ─── CLI ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="EVolvAI Physics-Informed TCN-VAE trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs",       type=int,   default=500,
                   help="Training epochs")
    p.add_argument("--batch",        type=int,   default=64,
                   help="Batch size")
    p.add_argument("--lr",           type=float, default=1e-3,
                   help="Adam initial learning rate")
    p.add_argument("--anneal",       type=int,   default=100,
                   help="KLD beta anneal epochs (0→1)")
    p.add_argument("--phys-anneal",  type=int,   default=150,
                   help="Physics lambda anneal epochs (0→full)")
    p.add_argument("--phys-start",   type=float, default=0.0,
                   help="Physics lambda starting fraction")
    p.add_argument("--clip",         type=float, default=1.0,
                   help="Gradient clip L2 norm")
    p.add_argument("--kld-max",      type=float, default=1.0,
                   help="Final KLD weight β")
    p.add_argument("--lr-step",      type=int,   default=150,
                   help="StepLR: reduce LR every N epochs")
    p.add_argument("--lr-gamma",     type=float, default=0.5,
                   help="StepLR: LR multiplicative factor")
    p.add_argument("--seed",         type=int,   default=42,
                   help="Global RNG seed")
    p.add_argument("--output",       type=str,   default="output",
                   help="Output directory for checkpoint + scenarios")
    p.add_argument("--log-every",    type=int,   default=25,
                   help="Print progress every N epochs")
    p.add_argument("--no-scenarios", action="store_true",
                   help="Skip counterfactual generation after training")
    return p.parse_args()


# ─── Training loop ────────────────────────────────────────────────────────────

def train(args):
    import torch
    import torch.optim as optim
    from generative_core import config as CFG
    from generative_core.data_loader import get_dataloader
    from generative_core.models import GenerativeCounterfactualVAE, vae_loss_function
    from generative_core.physics_loss import LinDistFlowLoss

    # Push CLI overrides into config
    CFG.EPOCHS        = args.epochs
    CFG.BATCH_SIZE    = args.batch
    CFG.LEARNING_RATE = args.lr

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[train] Device        : {device}")
    print(f"[train] Epochs        : {args.epochs}")
    print(f"[train] Batch         : {args.batch}   LR : {args.lr}")
    print(f"[train] KLD  anneal   : β 0 → {args.kld_max} over {args.anneal} epochs")
    print(f"[train] Phys anneal   : λ 0 → 1.0 over {args.phys_anneal} epochs")

    # DataLoader — automatically uses real parquet or synthetic fallback
    loader = get_dataloader(batch_size=args.batch)

    model          = GenerativeCounterfactualVAE().to(device)
    optimizer      = optim.Adam(model.parameters(), lr=args.lr)
    physics_engine = LinDistFlowLoss(device)
    scheduler      = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Parameters    : {total_params:,}")
    print(f"[model] TCN channels  : {CFG.TCN_CHANNELS}")
    print(f"[model] Latent dim    : {CFG.LATENT_DIM}")
    print(f"[model] Cond dim      : {CFG.COND_DIM}\n")

    # Store base physics lambdas before annealing modifies them
    _lv = CFG.LAMBDA_VOLT
    _lt = CFG.LAMBDA_THERMAL
    _lx = CFG.LAMBDA_XFMR

    history = []
    t_start = time.time()

    model.train()
    for epoch in range(1, args.epochs + 1):

        # KLD annealing: β 0 → kld_max
        beta = min(1.0, epoch / args.anneal) * args.kld_max
        CFG.KLD_WEIGHT = beta

        # Physics annealing: λ phys_start → 1.0
        lam  = args.phys_start + (1.0 - args.phys_start) * min(
            1.0, epoch / args.phys_anneal,
        )
        CFG.LAMBDA_VOLT    = _lv * lam
        CFG.LAMBDA_THERMAL = _lt * lam
        CFG.LAMBDA_XFMR    = _lx * lam

        epoch_loss = epoch_phys = 0.0
        n_batches  = 0

        for x, cond in loader:
            # x    : [B, NUM_FEATURES, SEQ_LEN]  (already channel-first from dataset)
            # cond : [B, COND_DIM]               (dynamic per-date condition)
            x    = x.to(device)
            cond = cond.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(x, cond)

            ev_demand = recon[:, :CFG.NUM_NODES, :].permute(0, 2, 1)  # [B, 24, 32]
            pen_v, pen_therm, pen_xfmr = physics_engine(ev_demand)
            phys = (CFG.LAMBDA_VOLT    * pen_v
                  + CFG.LAMBDA_THERMAL * pen_therm
                  + CFG.LAMBDA_XFMR   * pen_xfmr)

            loss = vae_loss_function(recon, x, mu, logvar, phys)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_phys += phys.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_phys = epoch_phys / max(n_batches, 1)
        history.append(avg_loss)

        if epoch % args.log_every == 0 or epoch == 1:
            elapsed = (time.time() - t_start) / 60
            eta     = (elapsed / epoch) * (args.epochs - epoch)
            lr_now  = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch:>4}/{args.epochs}  "
                f"loss={avg_loss:.5f}  phys={avg_phys:.5f}  "
                f"β={beta:.2f}  λ={lam:.2f}  lr={lr_now:.1e}  "
                f"elapsed={elapsed:.1f}min  ETA={eta:.1f}min"
            )
            drive_dir = os.path.join(args.output, "..", "google_drive_sync")
            os.makedirs(drive_dir, exist_ok=True)
            bkp = os.path.join(drive_dir, f"gcvae_model_ep{epoch}.pt")
            torch.save(model.state_dict(), bkp)

    total_min = (time.time() - t_start) / 60
    print(f"\n[train] Finished in {total_min:.1f} min ✓")
    return model, device, history


# ─── Save checkpoint + generate scenarios ────────────────────────────────────

def save_and_generate(model, device, history, args):
    import torch
    from generative_core import config as CFG

    os.makedirs(args.output, exist_ok=True)

    # ── Checkpoint ────────────────────────────────────────────────────────
    ckpt = os.path.join(args.output, "gcvae_model.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"\n[save] Checkpoint → {ckpt}")

    # ── Loss CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output, "training_loss.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,avg_loss\n")
        for i, v in enumerate(history, 1):
            f.write(f"{i},{v:.6f}\n")
    print(f"[save] Loss history → {csv_path}")

    if args.no_scenarios:
        return

    # ── Counterfactual generation ─────────────────────────────────────────
    print("\n[generate] Counterfactual scenarios…")
    model.eval()

    for name, spec in CFG.SCENARIOS.items():
        with torch.no_grad():
            z    = torch.randn(1, CFG.LATENT_DIM, device=device)
            cond = torch.tensor([spec["condition"]], dtype=torch.float32, device=device)
            out  = model.decode(z, cond)                # [1, NUM_FEATURES, 24]

            demand = out[:, :CFG.NUM_NODES, :].squeeze(0).permute(1, 0).cpu().numpy()
            # shape: [24, 32]

        path     = os.path.join(args.output, f"{name}.npy")
        np.save(path, demand)
        zero_pct = (demand == 0).mean() * 100
        flag     = "✅" if zero_pct < 30 else "⚠️ "
        print(
            f"  {flag} [{name:30s}]  shape={demand.shape}  "
            f"range=[{demand.min():.3f}, {demand.max():.3f}]  "
            f"zeros={zero_pct:.1f}%"
        )


# ─── Quick post-training quality check ───────────────────────────────────────

def quick_report(args):
    scenarios = [
        "extreme_winter_storm", "summer_peak",
        "full_electrification", "extreme_winter_v2", "rush_hour_gridlock",
    ]
    print("\n" + "=" * 62)
    print("  Post-training Quality Check")
    print("=" * 62)

    zero_pcts = []
    for name in scenarios:
        p = os.path.join(args.output, f"{name}.npy")
        if not os.path.exists(p):
            continue
        a = np.load(p)
        z = (a == 0).mean() * 100
        zero_pcts.append(z)
        flag = "✅" if z < 30 else "🟡" if z < 50 else "🔴"
        print(f"  {flag} {name:32s}  zeros={z:.1f}%  "
              f"range=[{a.min():.3f}, {a.max():.3f}]")

    if zero_pcts:
        avg_z = np.mean(zero_pcts)
        print(f"\n  Avg zeros : {avg_z:.1f}%")
        if avg_z < 20:
            print("  🟢  HEALTHY — ready for longer GPU run or handoff")
        elif avg_z < 40:
            print("  🟡  PARTIAL — run more epochs")
        else:
            print("  🔴  COLLAPSE — both annealings are ON; try more epochs first")
    print("  → Run python3 tester.py for the full report.")
    print("=" * 62)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  EVolvAI — Physics-Informed TCN-VAE  |  train.py")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("=" * 62)

    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    args = get_args()

    try:
        from generative_core import config as CFG
        from generative_core.models import GenerativeCounterfactualVAE, vae_loss_function
        from generative_core.physics_loss import LinDistFlowLoss
        from generative_core.data_loader import get_dataloader
        import torch
        print("[init] All imports OK ✓")
        print(f"[init] LATENT_DIM={CFG.LATENT_DIM}  TCN={CFG.TCN_CHANNELS}  "
              f"COND_DIM={CFG.COND_DIM}  DECODER={CFG.DECODER_HIDDEN}")
        parquet_exists = os.path.isfile(CFG.DATA_PATH)
        src = "real parquet" if parquet_exists else "synthetic fallback"
        print(f"[data] Source: {src}  ({'✅' if parquet_exists else '⚠  run preprocess.py first for real data'})")
    except ImportError as e:
        print(f"\n[error] Import failed: {e}")
        print("  Run: pip install torch numpy pandas pyarrow scipy")
        sys.exit(1)

    model, device, history = train(args)
    save_and_generate(model, device, history, args)
    quick_report(args)

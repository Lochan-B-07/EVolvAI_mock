"""
train.py — EVolvAI standalone training script
==============================================
Run from repo root on any machine with Python + PyTorch:

    python3 train.py                          # default 500 epochs
    python3 train.py --epochs 1000            # full run
    python3 train.py --epochs 50 --days 500   # quick test
    python3 train.py --help                   # all options

Works on CPU or GPU automatically.
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
    p.add_argument("--epochs",      type=int,   default=500,   help="Training epochs")
    p.add_argument("--batch",       type=int,   default=64,    help="Batch size")
    p.add_argument("--lr",          type=float, default=1e-3,  help="Adam learning rate")
    p.add_argument("--days",        type=int,   default=2000,  help="Synthetic training days (min 2000 recommended)")
    p.add_argument("--anneal",      type=int,   default=100,   help="KLD anneal epochs (β: 0→1)")
    p.add_argument("--phys-anneal", type=int,   default=150,   help="Physics lambda anneal epochs (λ: 0→full). Runs alongside KLD anneal.")
    p.add_argument("--phys-start",  type=float, default=0.0,   help="Physics lambda starting fraction (0=off, 0.1=10%% of full)")
    p.add_argument("--clip",        type=float, default=1.0,   help="Gradient clip norm")
    p.add_argument("--kld-max",     type=float, default=1.0,   help="Final KLD weight β")
    p.add_argument("--seed",        type=int,   default=42,    help="RNG seed")
    p.add_argument("--output",      type=str,   default="output", help="Output directory")
    p.add_argument("--log-every",   type=int,   default=25,    help="Print every N epochs")
    p.add_argument("--no-scenarios",action="store_true",       help="Skip counterfactual generation")
    return p.parse_args()


# ─── Lochan EV Scenario Generator (fast vectorised port) ─────────────────────
# Mirrors InputParameters_ScenerioGenerator.m exactly.

SG_TOTAL_EVS      = 100
SG_PEN_LEVEL      = 65
SG_TIME_INTERVALS = [0, 3, 7, 11, 14, 17, 19, 21, 24]
SG_CAR_DIST_RAW   = [6, 0, 5,  7, 17, 26, 22, 17]
SG_BC             = [22, 32, 40, 60]   # kWh
SG_CHARGING_POWER = 7.0                # kW
SG_CHARGERS_COUNT = 10
SG_MAX_WAIT_SEC   = 15 * 60           # seconds


def _car_dist():
    return [round(v * SG_PEN_LEVEL * SG_TOTAL_EVS / 10000) for v in SG_CAR_DIST_RAW]


def lochan_daily_demand_kw(num_nodes=32, rng=None):
    """
    Fast Python port of GenerateRandomSchedule_new_ScenerioGenerator.m.
    Returns shape [24, num_nodes] in kW.
    """
    if rng is None:
        rng = np.random.default_rng()

    car_dist     = _car_dist()
    time_slots_t = 24 * 3600          # total seconds in a day
    sec_per_15m  = 15 * 60

    power_sec = np.zeros(time_slots_t, dtype=np.float32)

    for i, num_cars in enumerate(car_dist):
        if num_cars == 0:
            continue
        t_start = SG_TIME_INTERVALS[i] * 4          # 15-min slot
        t_end   = SG_TIME_INTERVALS[i + 1] * 4
        n_slots = t_end - t_start
        if n_slots < 1:
            continue

        # Vectorised multinomial distribution (vs. slow while-loop in MATLAB)
        slots_cars = rng.multinomial(num_cars, np.ones(n_slots) / n_slots)

        for offset, noc in enumerate(slots_cars):
            if noc == 0:
                continue
            slot_abs = t_start + offset

            for _ in range(int(noc)):
                bc       = float(rng.choice(SG_BC))
                soc_act  = rng.integers(5, 81)
                soc_max  = rng.integers(int(soc_act) + 5, 101)
                e_needed = (soc_max - soc_act) * bc * 3600 / 100
                duration = int(e_needed / SG_CHARGING_POWER)

                ev_start = int(slot_abs * sec_per_15m + rng.integers(0, sec_per_15m + 1))
                ev_start = min(ev_start, time_slots_t - 1)
                ev_end   = min(ev_start + duration, time_slots_t)

                if power_sec[ev_start] < SG_CHARGERS_COUNT * SG_CHARGING_POWER:
                    power_sec[ev_start:ev_end] += SG_CHARGING_POWER

    hourly_kw = power_sec.reshape(24, 3600).mean(axis=1)
    weights   = rng.dirichlet(np.ones(num_nodes) * 0.8).astype(np.float32)
    return (hourly_kw[:, None] * weights[None, :]).astype(np.float32)


def build_dataset(n_days, num_nodes, seed, batch_size):
    """Generate n_days in RAM, normalise, return a DataLoader."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from generative_core import config as CFG

    print(f"[data] Generating {n_days} synthetic days (seed={seed})…")
    t0  = time.time()
    rng = np.random.default_rng(seed)

    demand  = np.stack(
        [lochan_daily_demand_kw(num_nodes=num_nodes, rng=rng) for _ in range(n_days)],
        axis=0,
    )  # [N, 24, nodes]

    weather = rng.uniform(-10, 40,
                          (n_days, 24, CFG.NUM_WEATHER_FEATURES)).astype(np.float32)

    def znorm(a):
        std = a.std()
        return (a - a.mean()) / (std + 1e-8) if std > 1e-8 else a - a.mean()

    data = np.concatenate([znorm(demand), znorm(weather)], axis=-1).astype(np.float32)
    dt   = time.time() - t0

    print(f"[data] Done in {dt:.1f}s  shape={data.shape}  "
          f"RAM={data.nbytes/1e6:.1f} MB")
    print(f"[data] kW range [{demand.min():.2f}, {demand.max():.2f}]  "
          f"mean={demand.mean():.3f}")

    tensor  = torch.from_numpy(data)                            # [N, 24, features]
    dataset = TensorDataset(tensor)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"[data] {len(dataset)} samples  {len(loader)} batches/epoch")
    return loader


# ─── Training ─────────────────────────────────────────────────────────────────
def train(args):
    import torch
    import torch.optim as optim
    from generative_core import config as CFG
    from generative_core.models import GenerativeCounterfactualVAE, vae_loss_function
    from generative_core.physics_loss import LinDistFlowLoss

    # Push CLI args into config
    CFG.EPOCHS        = args.epochs
    CFG.BATCH_SIZE    = args.batch
    CFG.LEARNING_RATE = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[train] Device : {device}")
    print(f"[train] Epochs : {args.epochs}  Batch : {args.batch}  "
          f"LR : {args.lr}  Days : {args.days}")
    print(f"[train] KLD   annealing : 0 → {args.kld_max} over {args.anneal} epochs")
    print(f"[train] Phys  annealing : 0%% → 100%% over {args.phys_anneal} epochs")
    if args.days < 500:
        print(f"[train] ⚠️  Only {args.days} days → {args.days//args.batch} batches/epoch. Recommend --days 2000+")
    print()


    loader = build_dataset(args.days, CFG.NUM_NODES, args.seed, args.batch)

    model          = GenerativeCounterfactualVAE().to(device)
    optimizer      = optim.Adam(model.parameters(), lr=args.lr)
    physics_engine = LinDistFlowLoss(device)
    scheduler      = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)

    baseline_cond = torch.tensor(
        CFG.BASELINE_CONDITION, dtype=torch.float32, device=device
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Parameters : {total_params:,}")

    history = []
    t_start = time.time()

    # Store base physics lambdas from config
    _base_volt    = CFG.LAMBDA_VOLT
    _base_thermal = CFG.LAMBDA_THERMAL
    _base_xfmr    = CFG.LAMBDA_XFMR

    model.train()
    for epoch in range(1, args.epochs + 1):

        # ── KLD annealing: β ramps 0 → kld_max over `anneal` epochs ──────
        beta = min(1.0, epoch / args.anneal) * args.kld_max
        CFG.KLD_WEIGHT = beta

        # ── Physics annealing: λ ramps phys_start → 1.0 over phys_anneal ─
        # This stops the decoder from collapsing to zeroes to avoid penalties.
        phys_scale = args.phys_start + (1.0 - args.phys_start) * min(
            1.0, epoch / args.phys_anneal
        )
        CFG.LAMBDA_VOLT    = _base_volt    * phys_scale
        CFG.LAMBDA_THERMAL = _base_thermal * phys_scale
        CFG.LAMBDA_XFMR    = _base_xfmr    * phys_scale

        epoch_loss = epoch_phys = 0.0
        n_batches  = 0

        for (batch,) in loader:
            x    = batch.permute(0, 2, 1).to(device)      # [B, features, 24]
            cond = baseline_cond.unsqueeze(0).expand(x.size(0), -1)

            optimizer.zero_grad()
            recon, mu, logvar = model(x, cond)

            ev_demand = recon[:, :CFG.NUM_NODES, :].permute(0, 2, 1)   # [B, 24, 32]
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
            print(f"  Epoch {epoch:>4}/{args.epochs}  "
                  f"loss={avg_loss:.5f}  phys={avg_phys:.5f}  "
                  f"β={beta:.2f}  λ={phys_scale:.2f}  lr={lr_now:.1e}  "
                  f"elapsed={elapsed:.1f}min  ETA={eta:.1f}min")

    total_min = (time.time() - t_start) / 60
    print(f"\n[train] Finished in {total_min:.1f} min ✓")
    return model, device, history


# ─── Save + Generate ──────────────────────────────────────────────────────────
def save_and_generate(model, device, history, args):
    import torch
    from generative_core import config as CFG

    os.makedirs(args.output, exist_ok=True)
    ckpt = os.path.join(args.output, "gcvae_model.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"\n[save] Checkpoint → {ckpt}")

    # Save loss history as CSV
    loss_csv = os.path.join(args.output, "training_loss.csv")
    with open(loss_csv, "w") as f:
        f.write("epoch,avg_loss\n")
        for i, v in enumerate(history, 1):
            f.write(f"{i},{v:.6f}\n")
    print(f"[save] Loss history → {loss_csv}")

    if args.no_scenarios:
        return

    print("\n[generate] Counterfactual scenarios…")
    model.eval()
    for name, spec in CFG.SCENARIOS.items():
        with torch.no_grad():
            z    = torch.randn(1, CFG.LATENT_DIM, device=device)
            cond = torch.tensor([spec["condition"]], dtype=torch.float32, device=device)
            out  = model.decode(z, cond)
            demand = out[:, :CFG.NUM_NODES, :].squeeze(0).permute(1, 0).cpu().numpy()

        path = os.path.join(args.output, f"{name}.npy")
        np.save(path, demand)
        zero_pct = (demand == 0).mean() * 100
        flag = "✅" if zero_pct < 30 else "⚠️ "
        print(f"  {flag} [{name:30s}]  shape={demand.shape}  "
              f"range=[{demand.min():.3f}, {demand.max():.3f}]  "
              f"zeros={zero_pct:.1f}%")


# ─── Inline quality report (same logic as tester.py) ─────────────────────────
def quick_report(args):
    print("\n" + "=" * 62)
    print("  Post-training Quality Check")
    print("=" * 62)

    scenarios = [
        "extreme_winter_storm", "summer_peak",
        "full_electrification", "extreme_winter_v2", "rush_hour_gridlock",
    ]

    arrays  = {}
    for name in scenarios:
        p = os.path.join(args.output, f"{name}.npy")
        if os.path.exists(p):
            arrays[name] = np.load(p)

    if not arrays:
        print("  No .npy files found — skipping report.")
        return

    zero_pcts = []
    for name, a in arrays.items():
        z = (a == 0).mean() * 100
        zero_pcts.append(z)
        flag = "✅" if z < 30 else "🟡" if z < 50 else "🔴"
        print(f"  {flag} {name:32s}  zeros={z:.1f}%  "
              f"range=[{a.min():.3f}, {a.max():.3f}]")

    avg_z = np.mean(zero_pcts)
    print(f"\n  Avg zeros : {avg_z:.1f}%")
    if avg_z < 20:
        print("  🟢  HEALTHY — ready for full 1000-epoch GPU run")
    elif avg_z < 40:
        print("  🟡  PARTIAL — run more epochs")
    else:
        print("  🔴  COLLAPSE — check KLD annealing + reduce physics λ")
    print("  → Run python3 tester.py for the full report anytime.")
    print("=" * 62)


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 62)
    print("  EVolvAI — Physics-Informed TCN-VAE Training")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("=" * 62)

    # Ensure repo root is on path (works whether run from root or subdir)
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    args = get_args()

    # Validate imports before spending time on data gen
    try:
        from generative_core import config as CFG          # noqa: F401
        from generative_core.models import (               # noqa: F401
            GenerativeCounterfactualVAE, vae_loss_function
        )
        from generative_core.physics_loss import LinDistFlowLoss  # noqa: F401
        import torch                                        # noqa: F401
        print("[init] All imports OK ✓")
        print(f"[init] NUM_NODES={CFG.NUM_NODES}  LATENT_DIM={CFG.LATENT_DIM}  "
              f"COND_DIM={CFG.COND_DIM}")
    except ImportError as e:
        print(f"\n[error] Import failed: {e}")
        print("  Make sure you are running from the repo root and requirements are installed.")
        print("  Run:  pip install torch numpy pandas pyarrow scipy")
        sys.exit(1)

    model, device, history = train(args)
    save_and_generate(model, device, history, args)
    quick_report(args)

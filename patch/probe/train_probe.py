"""
Probe frozen per-layer hidden states for event/non-event separability.

The npz (from HiddenStateCollector) holds one array per layer ('h_layer_8', ...)
plus shared y / vid / task. For each layer this trains a fresh LINEAR probe and a
2-layer MLP probe (grouped-by-video split) and reports AUROC, so you can see
whether any intermediate layer separates events better than the final layer
(which MMDuet's head reads, ~0.69-0.75).

Decision:
  - some middle layer AUROC >> final-layer AUROC  -> the firing head reads the wrong
        layer; cheap fix = read the better layer (no representation change).
  - all layers plateau (~0.75) and MLP ~= linear  -> signal not accessibly in any
        layer's last-token state -> representational fix needed.
  - MLP >> linear at some layer                    -> nonlinear readout helps there.

CRITICAL: grouped split by VIDEO (adjacent frames are near-identical; frame-level
split leaks and inflates AUROC).

Usage:
  python train_probe.py --npz hstates/hstates.npz --epochs 30
"""
import argparse
import numpy as np
import torch
import torch.nn as nn


def auroc(scores, labels):
    s = np.asarray(scores); y = np.asarray(labels)
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = y.sum(); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


class Probe(nn.Module):
    def __init__(self, d_in, d_hidden=512, nonlinear=True):
        super().__init__()
        if nonlinear:
            self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(),
                                     nn.Linear(d_hidden, 2))
        else:
            self.net = nn.Linear(d_in, 2)

    def forward(self, x):
        return self.net(x)


def train_one(Xtr, ytr, Xva, yva, nonlinear, epochs, device, pos_weight):
    model = Probe(Xtr.shape[1], nonlinear=nonlinear).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    w = torch.tensor([1.0, pos_weight], device=device)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    bs, n, best = 4096, len(Xtr_t), 0.0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            sc = torch.softmax(model(Xva_t), dim=-1)[:, 1].cpu().numpy()
        best = max(best, auroc(sc, yva))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    y = d["y"].astype(np.int64); vid = d["vid"]
    layer_keys = [k for k in d.files if k.startswith("h_layer_")]
    layer_keys = sorted(layer_keys, key=lambda k: int(k.split("_")[-1]))
    print(f"frames={len(y)}  event_frac={y.mean():.3f}  videos={len(set(vid))}  "
          f"layers={layer_keys}")

    # grouped split by video (shared across layers for fair comparison)
    rng = np.random.default_rng(args.seed)
    vids = np.array(sorted(set(vid))); rng.shuffle(vids)
    n_val = max(1, int(len(vids) * args.val_frac))
    val_vids = set(vids[:n_val])
    val_mask = np.array([v in val_vids for v in vid])
    ytr, yva = y[~val_mask], y[val_mask]
    pos_weight = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"train {len(ytr)} ({ytr.mean():.3f}) / val {len(yva)} ({yva.mean():.3f}); "
          f"{len(val_vids)} val videos; device={device}\n")

    print(f"  {'layer':10s} {'linear':>8s} {'MLP':>8s}")
    results = {}
    for k in layer_keys:
        H = d[k].astype(np.float32)
        Xtr, Xva = H[~val_mask], H[val_mask]
        lin = train_one(Xtr, ytr, Xva, yva, False, args.epochs, device, pos_weight)
        mlp = train_one(Xtr, ytr, Xva, yva, True, args.epochs, device, pos_weight)
        results[k] = (lin, mlp)
        print(f"  {k:10s} {lin:8.3f} {mlp:8.3f}")

    print("\n" + "=" * 56)
    best_layer = max(results, key=lambda k: max(results[k]))
    bl, bm = results[best_layer]
    final_key = layer_keys[-1]
    fl, fm = results[final_key]
    print(f"best layer: {best_layer}  (linear={bl:.3f} mlp={bm:.3f})")
    print(f"final layer ({final_key}): linear={fl:.3f} mlp={fm:.3f}")
    print(f"lift of best over final: {max(bl,bm) - max(fl,fm):+.3f}")
    print("-" * 56)
    print("  best >> final  -> firing head reads wrong layer (cheap fix)")
    print("  all plateau    -> representational fix needed")
    print("  MLP >> linear  -> nonlinear readout helps at that layer")
    print("=" * 56)


if __name__ == "__main__":
    main()
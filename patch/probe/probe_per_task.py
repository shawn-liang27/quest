"""
Per-task event-separability of the firing representation.

Trains ONE probe (pooled across tasks, grouped-by-video split) on a chosen layer,
then reports val AUROC stratified by task type. Tests hypothesis #2 (onset vs presence):

  - presence-type tasks (counting, grounding: "is the object there")
        -> representation should separate events WELL
  - onset-type tasks (instant_event_alert, semantic_alert: "is it happening NOW")
        -> representation should separate events POORLY

If counting/grounding >> alert/semantic -> representation encodes presence not onset
   -> onset/change features are the fix, targeted at onset-type tasks.
If flat across tasks -> weakness is uniform -> points to untrained-representation (#4).

Pooled training (not per-task) because per-task event counts are thin; we only
STRATIFY THE EVAL by task, which is the apples-to-apples per-task ceiling.

Usage:
  python probe_per_task.py --npz .../hstates.npz --layer h_layer_20 --epochs 40
"""
import argparse
import numpy as np
import torch
import torch.nn as nn


def auroc(scores, labels):
    s = np.asarray(scores); y = np.asarray(labels)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s); ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = y.sum(); n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


class Probe(nn.Module):
    def __init__(self, d, nonlinear=False, h=512):
        super().__init__()
        self.net = (nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, 2))
                    if nonlinear else nn.Linear(d, 2))
    def forward(self, x):
        return self.net(x)


# rough grouping for reading the result (onset-critical vs presence-critical)
ONSET_TASKS = {"instant_event_alert", "semantic_condition_alert", "event_narration",
               "sequential_step_instruction"}
PRESENCE_TASKS = {"static_object_counting", "dedup_counting", "cumulative_counting",
                  "explicit_target_grounding", "realtime_state_monitor"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--layer", default="h_layer_20")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nonlinear", action="store_true")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    y = d["y"].astype(np.int64); vid = d["vid"]
    if "task" not in d.files:
        raise SystemExit("npz has no 'task' array; cannot stratify by task")
    task = d["task"].astype(str)
    H = d[args.layer].astype(np.float32)
    print(f"layer={args.layer}  frames={len(y)}  event_frac={y.mean():.3f}  "
          f"videos={len(set(vid))}  tasks={len(set(task))}")

    # grouped split by video
    rng = np.random.default_rng(args.seed)
    vids = np.array(sorted(set(vid))); rng.shuffle(vids)
    val = set(vids[:max(1, int(len(vids) * args.val_frac))])
    vm = np.array([v in val for v in vid])
    Xtr, ytr = H[~vm], y[~vm]
    Xva, yva, tva = H[vm], y[vm], task[vm]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_w = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))

    model = Probe(H.shape[1], args.nonlinear).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w], device=device))
    Xt = torch.tensor(Xtr, device=device); yt = torch.tensor(ytr, device=device)
    bs = 4096
    for _ in range(args.epochs):
        model.train(); perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]; opt.zero_grad()
            lossf(model(Xt[idx]), yt[idx]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        sc = torch.softmax(model(torch.tensor(Xva, device=device)), dim=-1)[:, 1].cpu().numpy()

    print(f"\noverall val AUROC: {auroc(sc, yva):.3f}\n")
    print(f"  {'task':30s} {'n':>6s} {'events':>7s} {'AUROC':>7s}  type")
    rows = []
    for tk in sorted(set(tva)):
        m = tva == tk
        a = auroc(sc[m], yva[m])
        typ = "onset" if tk in ONSET_TASKS else "presence" if tk in PRESENCE_TASKS else "?"
        rows.append((tk, int(m.sum()), int(yva[m].sum()), a, typ))
        print(f"  {tk:30s} {int(m.sum()):6d} {int(yva[m].sum()):7d} {a:7.3f}  {typ}")

    # group means
    def grp(typ):
        vals = [a for _, _, ev, a, t in rows if t == typ and ev > 0 and not np.isnan(a)]
        return np.mean(vals) if vals else float("nan")
    print(f"\n  mean AUROC  onset-type tasks   : {grp('onset'):.3f}")
    print(f"  mean AUROC  presence-type tasks : {grp('presence'):.3f}")
    print("\n  presence >> onset -> representation encodes presence not onset (#2)")
    print("        -> build onset/change features, target onset-type tasks")
    print("  flat              -> uniform weakness -> untrained representation (#4)")


if __name__ == "__main__":
    main()
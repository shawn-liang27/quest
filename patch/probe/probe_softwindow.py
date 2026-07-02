"""
Label-ceiling test: is 0.79 a REPRESENTATION ceiling or an EVAL-STRICTNESS (label) ceiling?

Hypothesis: AUROC may cap at 0.79 not because the representation is weak, but because
frame-level "event" labels are fuzzy near boundaries (a trigger is a moment, but frames
±a few seconds are ambiguous). AUROC is depressed precisely by mislabeled near-boundary
frames. If so, no finetuning passes 0.79 — it's the label, not the representation.

Two probes of this, both reuse the existing npz (needs per-frame 'time'; if absent, see
--time_from_order note):

  A) SOFTENED / WIDENED positive window: relabel events with tolerance W (e.g. 1,3,5,7s)
     and re-probe. If AUROC JUMPS toward 0.9 as W widens, the representation cleanly
     encodes "near an event" and 0.79@tol3 was frame-strictness.

  B) AUROC vs DISTANCE-FROM-EVENT-CENTER: among event frames, bucket by |t - nearest
     trigger|, and measure how confidently the probe scores them positive. If the probe
     is near-perfect AT CENTER (|d|<1s) and only fails on boundary frames (2-3s out),
     then the representation is fine for "fire once near center" and the ceiling is the
     boundary fuzz, not the representation.

Requires a per-frame 'time' array in the npz. The collector stores time inside debug,
but train_probe's npz may not have it. If 'time' is missing we reconstruct event-distance
is impossible -> the script will tell you to re-dump with time, OR you can pass a
companion pred jsonl (--pred_file) whose debug_data has per-frame time aligned 1-1 to npz
rows in the SAME order they were added.

Usage:
  python probe_softwindow.py --npz hstates.npz --gt_file metadata.jsonl \
      --pred_file base-pred.jsonl --layer h_layer_20 --windows 1,3,5,7
"""
import argparse, json
import numpy as np
import torch
import torch.nn as nn


def auroc(s, y):
    s = np.asarray(s); y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s); ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    np_pos = y.sum(); nn_ = len(y) - np_pos
    return float((ranks[y == 1].sum() - np_pos * (np_pos + 1) / 2) / (np_pos * nn_))


class Probe(nn.Module):
    def __init__(self, d, h=512, nonlinear=False):
        super().__init__()
        self.net = (nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, 2))
                    if nonlinear else nn.Linear(d, 2))
    def forward(self, x): return self.net(x)


def get_times(d, npz_path, pred_file, gt):
    """Return per-row time and per-row nearest-trigger distance, aligned to npz rows."""
    if "time" in d.files:
        t = d["time"].astype(np.float32)
    elif pred_file:
        # reconstruct in the SAME order the collector added rows: per video, per frame
        # with an h_layer_* key. We mirror dump_hidden_states.add_video ordering.
        vid = d["vid"]
        order_times = []
        fires_by_qid = {}
        for line in open(pred_file):
            if not line.strip(): continue
            r = json.loads(line)
            qid = r.get("question_id") or r.get("id")
            fires_by_qid[qid] = [fr.get("time") for fr in r.get("debug_data", [])
                                 if any(k.startswith("h_layer_") for k in fr)]
        # walk npz vids in order, popping times per video
        from collections import defaultdict, deque
        q = {k: deque(v) for k, v in fires_by_qid.items()}
        for v in vid:
            if v in q and q[v]:
                order_times.append(q[v].popleft())
            else:
                order_times.append(np.nan)
        t = np.array(order_times, dtype=np.float32)
        if np.isnan(t).any():
            raise SystemExit("time reconstruction misaligned (NaNs). Re-dump npz with a "
                             "'time' array for a reliable run.")
    else:
        raise SystemExit("npz has no 'time'. Re-dump storing per-frame time, or pass "
                         "--pred_file whose debug_data time aligns to npz row order.")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--gt_file", required=True)
    ap.add_argument("--pred_file", default=None,
                    help="used to recover per-frame time if npz lacks it")
    ap.add_argument("--layer", default="h_layer_20")
    ap.add_argument("--windows", default="1,3,5,7")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    vid = d["vid"]; H = d[args.layer].astype(np.float32)

    # gt triggers per video
    gt = {}
    for line in open(args.gt_file):
        if not line.strip(): continue
        r = json.loads(line)
        if r.get("audio_dependency") != "none":  # visual-only
            continue
        raw = r["ground_truth"]
        tr = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(tr, dict): tr = tr.get("triggers", [])
        gt[r["id"]] = [float(x["trigger_time_sec"]) for x in tr if "trigger_time_sec" in x]

    t = get_times(d, args.npz, args.pred_file, gt)
    # nearest-trigger distance per row
    dist = np.full(len(t), 1e9, dtype=np.float32)
    for i, v in enumerate(vid):
        gts = gt.get(v, [])
        if gts:
            dist[i] = min(abs(t[i] - g) for g in gts)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    vids = np.array(sorted(set(vid))); rng.shuffle(vids)
    val = set(vids[:max(1, int(len(vids) * args.val_frac))])
    vm = np.array([v in val for v in vid])

    print("=" * 60)
    print(f"LABEL-CEILING TEST  layer={args.layer}  frames={len(vid)}")
    print("=" * 60)

    # ---- A) widen the positive window, re-probe ----
    print("\nA) AUROC vs positive-window tolerance:")
    print(f"  {'window(s)':>10s} {'event_frac':>11s} {'AUROC':>7s}")
    for W in [float(x) for x in args.windows.split(",")]:
        y = (dist <= W).astype(np.int64)
        ytr, yva = y[~vm], y[vm]
        if ytr.sum() == 0 or yva.sum() == 0:
            print(f"  {W:10.1f}   (no positives in split)"); continue
        pos_w = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
        model = Probe(H.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w], device=device))
        Xt = torch.tensor(H[~vm], device=device); yt = torch.tensor(ytr, device=device)
        for _ in range(args.epochs):
            model.train(); perm = torch.randperm(len(Xt), device=device)
            for i in range(0, len(Xt), 4096):
                idx = perm[i:i+4096]; opt.zero_grad()
                lossf(model(Xt[idx]), yt[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            sc = torch.softmax(model(torch.tensor(H[vm], device=device)), -1)[:, 1].cpu().numpy()
        print(f"  {W:10.1f} {yva.mean():11.3f} {auroc(sc, yva):7.3f}")
    print("  JUMP toward 0.9 as window widens -> 0.79@tol3 was frame-strictness (label),")
    print("  not representation. FLAT -> representation truly caps here.")

    # ---- B) AUROC by distance-from-center, using the tol=3 probe ----
    print("\nB) probe confidence vs distance-from-event-center (tol=3 probe):")
    y3 = (dist <= 3.0).astype(np.int64)
    ytr = y3[~vm]
    pos_w = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    model = Probe(H.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w], device=device))
    Xt = torch.tensor(H[~vm], device=device); yt = torch.tensor(ytr, device=device)
    for _ in range(args.epochs):
        model.train(); perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), 4096):
            idx = perm[i:i+4096]; opt.zero_grad()
            lossf(model(Xt[idx]), yt[idx]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        sc = torch.softmax(model(torch.tensor(H[vm], device=device)), -1)[:, 1].cpu().numpy()
    dva = dist[vm]
    # mean probe score for event frames bucketed by |d|, vs non-event baseline
    print(f"  {'dist band':>12s} {'n':>6s} {'mean score':>11s}")
    neg_score = sc[dva > 10].mean() if (dva > 10).any() else float('nan')
    print(f"  {'non-event':>12s} {(dva>10).sum():6d} {neg_score:11.3f}")
    for lo, hi in [(0,1),(1,2),(2,3),(3,5),(5,10)]:
        m = (dva >= lo) & (dva < hi)
        if m.sum() == 0: continue
        print(f"  {f'{lo}-{hi}s':>12s} {int(m.sum()):6d} {sc[m].mean():11.3f}")
    print("  high score at 0-1s falling off by 2-3s -> probe nails CENTER, fails only at")
    print("  boundary -> representation fine for 'fire near center'; fix = windowed target.")
    print("=" * 60)


if __name__ == "__main__":
    main()
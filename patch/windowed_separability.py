"""
Windowed separability test.

Per-frame relevance_score separates events weakly (AUROC ~0.69). Question:
is the firing signal TEMPORAL — present across a window of frames but not at any
single frame? If aggregating the score over k consecutive frames lifts AUROC
substantially, the fix is sequential firing (accumulate evidence), NOT a better
per-frame representation.

For each frame t, build a windowed score from the per-frame relevance scores in
[t-k+1 .. t] (causal, like a real streaming detector) using several aggregators:
  - mean   : average score over the window (smoothing)
  - max    : peak score in the window
  - sum    : total score (CUSUM-like accumulation)
  - ewma   : exponentially-weighted moving average
Then label frame t as event/non-event (within tol of a GT trigger) and compute
AUROC for each aggregator and window size.

Compare windowed AUROC to per-frame AUROC (window=1):
  - big lift  -> signal is temporal -> sequential firing is the fix
  - no lift   -> signal is per-frame-limited -> representational fix needed

Usage:
  python diag_windowed_separability.py \
      --pred_file outputs/.../omnipro_visual-pred.jsonl \
      --gt_file   /mnt/data0/sgl57/data/omnipro/metadata.jsonl \
      --score_head relevance_score \
      --windows 1,3,5,9,15 --tolerance 3.0 --per_task
"""
import json, argparse
import numpy as np


def load_gt(path, visual_only=True):
    gt = {}
    for line in open(path):
        if not line.strip():
            continue
        row = json.loads(line)
        if visual_only and row.get("audio_dependency") != "none":
            continue
        raw = row["ground_truth"]
        trigs = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(trigs, dict):
            trigs = trigs.get("triggers", [])
        gt[row["id"]] = {
            "task": row["task"],
            "gt_times": [float(t["trigger_time_sec"]) for t in trigs
                         if "trigger_time_sec" in t],
        }
    return gt


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    allv = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum = sum(rank for rank, (_, lbl) in enumerate(allv, 1) if lbl == 1)
    n_pos, n_neg = len(pos), len(neg)
    u = rank_sum - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def windowed_scores(scores, k, agg, ewma_alpha=0.3):
    """Causal window of size k ending at each index."""
    out = []
    if agg == "ewma":
        e = None
        for s in scores:
            e = s if e is None else ewma_alpha * s + (1 - ewma_alpha) * e
            out.append(e)
        return out
    for i in range(len(scores)):
        w = scores[max(0, i - k + 1): i + 1]
        if agg == "mean":
            out.append(float(np.mean(w)))
        elif agg == "max":
            out.append(float(np.max(w)))
        elif agg == "sum":
            out.append(float(np.sum(w)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True)
    ap.add_argument("--gt_file", required=True)
    ap.add_argument("--score_head", default="relevance_score")
    ap.add_argument("--windows", default="1,3,5,9,15")
    ap.add_argument("--aggs", default="mean,max,sum,ewma")
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--per_task", action="store_true")
    args = ap.parse_args()

    windows = [int(w) for w in args.windows.split(",")]
    aggs = args.aggs.split(",")
    gt = load_gt(args.gt_file)

    # collect per-video ordered score series + event labels
    series = []   # list of (task, scores[list], labels[list])
    for line in open(args.pred_file):
        if not line.strip():
            continue
        d = json.loads(line)
        qid = d.get("question_id") or d.get("id")
        if qid not in gt:
            continue
        debug = d.get("debug_data")
        if not debug:
            continue
        debug = sorted(debug, key=lambda x: x.get("time", 0))
        scores = [fr.get(args.score_head, 0.0) for fr in debug]
        times = [fr.get("time", i) for i, fr in enumerate(debug)]
        gtt = gt[qid]["gt_times"]
        labels = [1 if any(abs(t - g) <= args.tolerance for g in gtt) else 0
                  for t in times]
        if any(labels) and not all(labels):   # need both classes
            series.append((gt[qid]["task"], scores, labels))

    if not series:
        raise SystemExit("No usable debug_data series found.")

    print("=" * 70)
    print(f"Windowed separability  (head={args.score_head}, tol={args.tolerance}s, "
          f"videos={len(series)})")
    print("=" * 70)
    print("\nAUROC by aggregator x window (window=1 = per-frame baseline):\n")

    header = "  agg     " + "".join(f"k={k:<6d}" for k in windows)
    print(header)
    best = {}
    for agg in aggs:
        row = f"  {agg:7s} "
        for k in windows:
            pos, neg = [], []
            for task, scores, labels in series:
                ws = windowed_scores(scores, k, agg)
                for s, l in zip(ws, labels):
                    (pos if l else neg).append(s)
            a = auc(pos, neg)
            row += f"{a:.3f}  "
            best[(agg, k)] = a
        print(row)

    # baseline = mean/per-frame (k=1 is identical for mean/max/sum)
    base = best.get(("mean", 1), float("nan"))
    top = max((v for v in best.values() if not np.isnan(v)), default=float("nan"))
    print(f"\n  per-frame AUROC (k=1): {base:.3f}")
    print(f"  best windowed AUROC  : {top:.3f}  "
          f"(lift {top-base:+.3f})")
    print("\nInterpretation:")
    print("  lift >= ~0.05-0.10  -> signal is TEMPORAL: sequential firing is the fix")
    print("  lift ~ 0            -> per-frame-limited: representational fix needed")

    if args.per_task:
        print("\nPer-task: per-frame vs best-windowed AUROC")
        tasks = sorted(set(t for t, _, _ in series))
        for task in tasks:
            ts = [(s, l) for tk, s, l in series if tk == task]
            # per-frame
            pf_pos = [sc for scores, labels in ts for sc, l in zip(scores, labels) if l]
            pf_neg = [sc for scores, labels in ts for sc, l in zip(scores, labels) if not l]
            pf = auc(pf_pos, pf_neg)
            # best window (use 'max' over largest window as a representative temporal agg)
            kbest = max(windows)
            bw_pos, bw_neg = [], []
            for scores, labels in ts:
                ws = windowed_scores(scores, kbest, "max")
                for s, l in zip(ws, labels):
                    (bw_pos if l else bw_neg).append(s)
            bw = auc(bw_pos, bw_neg)
            print(f"  {task:30s} per-frame={pf:.3f}  windowed(k={kbest},max)={bw:.3f}  "
                  f"lift={bw-pf:+.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
"""
Separability test for MMDuet's per-frame trigger head.

MMDuet logs, per frame, informative_score and relevance_score in res['debug_data'].
The firing decision thresholds the sum of the selected score_heads. This script
asks the question that decides whether a better firing head can even help:

    Do true-event frames have HIGHER trigger scores than non-event frames?

If yes (separable): the signal exists, the head just thresholds it badly ->
    a calibrated / sequential decision CAN improve precision (differentiator real).
If no (overlapping): no threshold separates events from non-events ->
    the failure is representational, a better head won't fix it.

Outputs: score distributions at event vs non-event frames, AUC (how separable),
and the precision-recall curve achievable by sweeping the threshold (i.e. the
operating-point curve the bimodal decision never exposes).

Usage:
  python diag_score_separability.py \
      --pred_file outputs/.../omnipro_visual-pred.jsonl \
      --gt_file   /mnt/data0/sgl57/data/omnipro/metadata.jsonl \
      --score_heads informative_score,relevance_score \
      --tolerance 3.0
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


def auc_from_scores(pos, neg):
    """AUROC via the rank statistic (Mann-Whitney). No sklearn dependency."""
    if not pos or not neg:
        return float("nan")
    allv = [(s, 1) for s in pos] + [(s, 0) for s in neg]
    allv.sort(key=lambda x: x[0])
    rank_sum = 0
    for rank, (_, lbl) in enumerate(allv, start=1):
        if lbl == 1:
            rank_sum += rank
    n_pos, n_neg = len(pos), len(neg)
    u = rank_sum - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def pr_curve(pos, neg, n_thresholds=50):
    """Precision/recall achievable by sweeping a threshold on the score."""
    alls = sorted(set(pos + neg))
    if len(alls) > n_thresholds:
        idx = np.linspace(0, len(alls) - 1, n_thresholds).astype(int)
        thresholds = [alls[i] for i in idx]
    else:
        thresholds = alls
    pos_a, neg_a = np.array(pos), np.array(neg)
    out = []
    for th in thresholds:
        tp = int((pos_a >= th).sum())
        fp = int((neg_a >= th).sum())
        fn = int((pos_a < th).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out.append((th, prec, rec))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True)
    ap.add_argument("--gt_file", required=True)
    ap.add_argument("--score_heads", default="informative_score,relevance_score",
                    help="comma list; summed per frame like MMDuet's decision")
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--per_task", action="store_true")
    args = ap.parse_args()

    heads = args.score_heads.split(",")
    gt = load_gt(args.gt_file)

    pos_scores, neg_scores = [], []   # event-frame scores vs non-event-frame scores
    by_task_pos, by_task_neg = {}, {}

    n_used = 0
    for line in open(args.pred_file):
        if not line.strip():
            continue
        d = json.loads(line)
        qid = d.get("question_id") or d.get("id")
        if qid not in gt:
            continue
        debug = d.get("debug_data")
        if not debug:
            continue   # this run didn't log per-frame scores
        n_used += 1
        gtt = gt[qid]["gt_times"]
        task = gt[qid]["task"]
        by_task_pos.setdefault(task, []); by_task_neg.setdefault(task, [])
        for fr in debug:
            t = fr.get("time")
            score = sum(fr.get(h, 0.0) for h in heads)
            is_event = any(abs(t - g) <= args.tolerance for g in gtt)
            if is_event:
                pos_scores.append(score); by_task_pos[task].append(score)
            else:
                neg_scores.append(score); by_task_neg[task].append(score)

    if n_used == 0:
        raise SystemExit("No debug_data found in pred file — re-run inference "
                         "with this script so res['debug_data'] is logged.")

    print("=" * 64)
    print(f"Trigger-score separability  (heads={heads}, tol={args.tolerance}s, "
          f"samples={n_used})")
    print("=" * 64)

    def stat(name, v):
        a = np.array(v)
        print(f"  {name:18s} n={len(v):7d}  mean={a.mean():.4f}  "
              f"median={np.median(a):.4f}  p90={np.percentile(a,90):.4f}")

    print("\nScore at frame type:")
    stat("EVENT frames", pos_scores)
    stat("NON-EVENT frames", neg_scores)

    auc = auc_from_scores(pos_scores, neg_scores)
    print(f"\nAUROC (event vs non-event separability): {auc:.4f}")
    print("  0.5 = no separation (head sees nothing) -> representational fix needed")
    print("  >0.8 = strongly separable -> signal exists, decision is the problem")

    print("\nAchievable precision/recall by sweeping the score threshold:")
    print(f"  {'thresh':>8s} {'precision':>10s} {'recall':>8s}")
    for th, p, r in pr_curve(pos_scores, neg_scores):
        if r > 0:  # skip dead thresholds
            print(f"  {th:8.3f} {p:10.3f} {r:8.3f}")
    print("  (If some threshold gives high precision at decent recall, a")
    print("   calibrated decision recovers it — the bimodal head just can't pick it.)")

    if args.per_task:
        print("\nPer-task AUROC:")
        for task in by_task_pos:
            a = auc_from_scores(by_task_pos[task], by_task_neg[task])
            print(f"  {task:30s} AUROC={a:.4f}  "
                  f"(event n={len(by_task_pos[task])})")
    print("=" * 64)


if __name__ == "__main__":
    main()
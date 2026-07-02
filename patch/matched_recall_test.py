"""
Matched-recall PR test: is re-prompting's precision gain a real decision improvement,
or just firing less (= an implicit threshold shift)?

Re-prompting raised precision (0.093->0.146), lowered recall (0.863->0.707), left probe
AUROC flat. This asks: is the suppression UNIFORM (same as raising the threshold -> adds
nothing) or SELECTIVE toward non-events (beats the threshold curve -> real decision gain)?

Method: per-frame score = sum of score_heads from debug_data. Sweep the threshold to get
the full precision-recall curve for BOTH runs, where:
    recall    = event-frames fired / total event-frames      (in [0,1])
    precision = event-frames fired / all-frames fired
Then compare the two CURVES, and read precision at matched recall points.

  reprompt curve ABOVE baseline curve  -> selective suppression (real gain)
  curves OVERLAP                       -> uniform suppression (= thresholding)

Frame is an event-frame if within --tol of a GT trigger.

Usage:
  python matched_recall_test.py --baseline base.jsonl --reprompt rep.jsonl \
      --gt_file /mnt/data0/sgl57/data/omnipro/metadata.jsonl \
      --score_heads informative_score,relevance_score --tol 3.0
"""
import json, argparse
import numpy as np


def load_gt(path, visual_only=True):
    gt = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if visual_only and r.get("audio_dependency") != "none":
            continue
        raw = r["ground_truth"]
        trigs = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(trigs, dict):
            trigs = trigs.get("triggers", [])
        gt[r["id"]] = [float(t["trigger_time_sec"]) for t in trigs if "trigger_time_sec" in t]
    return gt


def frame_scores_labels(pred_path, gt, score_heads, tol):
    scores, labels = [], []
    for line in open(pred_path):
        if not line.strip():
            continue
        d = json.loads(line)
        qid = d.get("question_id") or d.get("id")
        if qid not in gt:
            continue
        gts = gt[qid]
        for fr in d.get("debug_data", []):
            if "time" not in fr:
                continue
            s = 0.0; ok = False
            for h in score_heads:
                if h in fr and fr[h] is not None:
                    v = fr[h]
                    s += v[1] if isinstance(v, (list, tuple)) else float(v)
                    ok = True
            if not ok:
                continue
            t = float(fr["time"])
            labels.append(1 if any(abs(t - g) <= tol for g in gts) else 0)
            scores.append(s)
    return np.array(scores), np.array(labels)


def pr_curve(scores, labels):
    """Sweep threshold high->low. recall = TP/total_event_frames (in [0,1])."""
    n_pos = int(labels.sum())
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(n_pos, 1)
    return recall, precision


def precision_at_recall(recall, precision, target):
    """Best (max) precision among points with recall >= target."""
    mask = recall >= target
    if not mask.any():
        return float("nan")
    return float(precision[mask].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--reprompt", required=True)
    ap.add_argument("--gt_file", required=True)
    ap.add_argument("--score_heads", default="informative_score,relevance_score")
    ap.add_argument("--tol", type=float, default=3.0)
    args = ap.parse_args()

    heads = args.score_heads.split(",")
    gt = load_gt(args.gt_file)

    sb, yb = frame_scores_labels(args.baseline, gt, heads, args.tol)
    sr, yr = frame_scores_labels(args.reprompt, gt, heads, args.tol)
    if len(sb) == 0 or len(sr) == 0:
        raise SystemExit("no per-frame scores found; check debug_data has the score heads "
                         "and --score_heads matches their names.")

    rb, pb = pr_curve(sb, yb)
    rr, pr_ = pr_curve(sr, yr)

    print("=" * 64)
    print("MATCHED-RECALL PR TEST  (frame-level)")
    print("=" * 64)
    print(f"baseline: frames={len(sb)} event_frames={int(yb.sum())} ({yb.mean():.3f})")
    print(f"reprompt: frames={len(sr)} event_frames={int(yr.sum())} ({yr.mean():.3f})\n")

    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    def auc(r, p):
        idx = np.argsort(r); return float(trap(p[idx], r[idx]))
    print(f"AUC-PR (sweep):  baseline={auc(rb,pb):.4f}   reprompt={auc(rr,pr_):.4f}")
    print("  higher = better PR curve. reprompt>baseline => the CURVE moved (selective);")
    print("  ~equal => only the operating point differs (uniform = thresholding).\n")

    print("precision at matched recall (baseline vs reprompt):")
    print(f"  {'recall':>8s} {'base P':>8s} {'reprompt P':>11s}   verdict")
    any_selective = False
    for tr in [0.5, 0.6, 0.7, 0.71, 0.8, 0.86, 0.9]:
        bp = precision_at_recall(rb, pb, tr)
        rp = precision_at_recall(rr, pr_, tr)
        if np.isnan(bp) or np.isnan(rp):
            continue
        verdict = "reprompt better" if rp > bp + 0.005 else \
                  "baseline better" if bp > rp + 0.005 else "tie"
        if rp > bp + 0.005:
            any_selective = True
        print(f"  {tr:8.2f} {bp:8.3f} {rp:11.3f}   {verdict}")

    print()
    print("READ:")
    print("  reprompt P > baseline P at matched recall  -> SELECTIVE suppression")
    print("     (re-prompting fires less ON NON-EVENTS specifically; real decision gain,")
    print("      not reachable by thresholding the baseline).")
    print("  reprompt P ~= baseline P at every recall   -> UNIFORM suppression")
    print("     (re-prompting == raising the threshold; precision rose only because it")
    print("      fires less overall, NOT because it knows when to fire).")
    print("=" * 64)


if __name__ == "__main__":
    main()
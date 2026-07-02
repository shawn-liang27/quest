"""
Coverage-cap diagnostic for the long-trigger buckets.

MMDuet2 ingests frames sequentially up to a cap:  max_num_frames / fps seconds.
Any trigger AFTER that cap is unseeable -> a "miss" there is truncation, not
temporal decay. This script checks, per trigger-time bucket, how many GT
triggers fall beyond the model's coverage horizon, and whether the misses in
the 300-600s bucket are explained by the cap.

Also reports the duration distribution of the visual-only subset (longest video
etc.) so you know whether the benchmark even contains the horizons you're
claiming to test.

Set --fps and --max_frames to MMDuet2's actual inference settings.
  (from the inference code you inspected: fps=1, max_num_frames=400 -> cap=400s)

Usage:
  python diag_coverage_cap.py \
      --pred_file outputs/full_2/omnipro_visual-pred.jsonl \
      --gt_file   /mnt/data0/sgl57/data/omnipro/metadata.jsonl \
      --fps 1.0 --max_frames 400 --tolerance 5.0
"""
import json, argparse, math
from collections import defaultdict


def load_pred(path):
    preds = {}
    for line in open(path):
        if not line.strip():
            continue
        d = json.loads(line)
        qid = d.get("question_id") or d.get("id")
        preds[qid] = sorted(float(r["time"]) for r in d.get("model_response_list", [])
                            if r.get("role") == "assistant" and "time" in r)
    return preds


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
        times = [float(t["trigger_time_sec"]) for t in trigs
                 if "trigger_time_sec" in t]
        gt[row["id"]] = {"task": row["task"], "gt_times": times,
                         "duration": row.get("duration")}
    return gt


BUCKETS = [(0, 30), (30, 60), (60, 120), (120, 180), (180, 300), (300, 600), (600, 1e9)]


def bucket_of(t):
    for lo, hi in BUCKETS:
        if lo <= t < hi:
            return f"{lo}-{hi if hi < 1e9 else 'inf'}s"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True)
    ap.add_argument("--gt_file", default="/mnt/data0/sgl57/data/omnipro/metadata.jsonl")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max_frames", type=int, default=400)
    ap.add_argument("--tolerance", type=float, default=5.0)
    args = ap.parse_args()

    cap_sec = args.max_frames / args.fps
    preds = load_pred(args.pred_file)
    gt = load_gt(args.gt_file)

    # ---- duration distribution of the visual-only subset ----
    durs = sorted(v["duration"] for v in gt.values() if v["duration"])
    print("=" * 88)
    print("VISUAL-ONLY SUBSET — duration distribution")
    print("=" * 88)
    if durs:
        n = len(durs)
        print(f"  videos: {n}")
        print(f"  min/median/max duration: "
              f"{durs[0]:.0f}s / {durs[n//2]:.0f}s / {durs[-1]:.0f}s")
        print(f"  videos longer than cap ({cap_sec:.0f}s): "
              f"{sum(1 for d in durs if d > cap_sec)}")
        for thr in (180, 300, 400, 600):
            print(f"  videos > {thr}s: {sum(1 for d in durs if d > thr)}")

    # ---- per-bucket metrics ----
    print("\n" + "=" * 88)
    print(f"COVERAGE CAP = {cap_sec:.0f}s  (fps={args.fps}, max_frames={args.max_frames})")
    print("=" * 88)

    run = {qid: g for qid, g in gt.items() if qid in preds}
    print(f"  scored samples (in pred file): {len(run)}\n")

    per_bucket = defaultdict(lambda: {
        "trigs": 0, "beyond_cap": 0, "detected": 0,
        "missed_beyond": 0, "missed_within": 0,
        "total_preds": 0, "correct_preds": 0,
        "total_preds_within": 0, "correct_preds_within": 0
    })

    for qid, g in run.items():
        fires = preds.get(qid, [])
        
        # 1. Evaluate Ground Truths (Recall, Misses, Cap)
        for gt_t in g["gt_times"]:
            b = bucket_of(gt_t)
            pb = per_bucket[b]
            pb["trigs"] += 1
            beyond = gt_t > cap_sec
            if beyond:
                pb["beyond_cap"] += 1
                
            detected = any(abs(f - gt_t) <= args.tolerance for f in fires)
            if detected:
                pb["detected"] += 1
            else:
                if beyond:
                    pb["missed_beyond"] += 1   # miss explained by truncation
                else:
                    pb["missed_within"] += 1   # genuine miss (model saw it, didn't fire)

        # 2. Evaluate Predictions (Precision)
        for f in fires:
            b = bucket_of(f)
            pb = per_bucket[b]
            pb["total_preds"] += 1
            
            is_correct = any(abs(f - gt_t) <= args.tolerance for gt_t in g["gt_times"])
            if is_correct:
                pb["correct_preds"] += 1
                
            # Track predictions within the coverage cap for adjusted precision
            if f <= cap_sec:
                pb["total_preds_within"] += 1
                if is_correct:
                    pb["correct_preds_within"] += 1

    # Table Header
    print(f"  {'bucket':11s} {'trigs':>5s} {'>cap':>4s} {'preds':>5s} | "
          f"{'Rec':>5s} {'Prec':>5s} {'F1':>5s} | "
          f"{'m>cap':>5s} {'m<=cap':>6s} | "
          f"{'Rec_w':>5s} {'Prec_w':>6s} {'F1_w':>5s}")
    print("  " + "-" * 85)

    for lo, hi in BUCKETS:
        b = f"{lo}-{hi if hi < 1e9 else 'inf'}s"
        pb = per_bucket.get(b)
        # Skip empty buckets where there are neither GT triggers nor model predictions
        if not pb or (pb["trigs"] == 0 and pb["total_preds"] == 0):
            continue
            
        # Standard Metrics
        recall = pb["detected"] / pb["trigs"] if pb["trigs"] > 0 else 0.0
        precision = pb["correct_preds"] / pb["total_preds"] if pb["total_preds"] > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Within-Cap Metrics (excluding triggers/preds beyond the coverage cap)
        within_trigs = pb["trigs"] - pb["beyond_cap"]
        recall_w = pb["detected"] / within_trigs if within_trigs > 0 else float("nan")
        prec_w = pb["correct_preds_within"] / pb["total_preds_within"] if pb["total_preds_within"] > 0 else float("nan")
        
        if not math.isnan(recall_w) and not math.isnan(prec_w) and (prec_w + recall_w) > 0:
            f1_w = 2 * (prec_w * recall_w) / (prec_w + recall_w)
        else:
            f1_w = float("nan")

        # Helper to format NaN gracefully
        def fmt(val):
            return f"{val:5.2f}" if not math.isnan(val) else "  NaN"

        print(f"  {b:11s} {pb['trigs']:5d} {pb['beyond_cap']:4d} {pb['total_preds']:5d} | "
              f"{fmt(recall)} {fmt(precision)} {fmt(f1)} | "
              f"{pb['missed_beyond']:5d} {pb['missed_within']:6d} | "
              f"{fmt(recall_w)} {fmt(prec_w)} {fmt(f1_w)}")

    print("\nInterpretation:")
    print("  - '>cap'       = GT triggers the model NEVER ingested (truncation).")
    print("  - 'Rec/Prec/F1'= Standard metrics over all triggers and predictions in the bucket.")
    print("  - 'm>cap'      = Misses explained entirely by truncation.")
    print("  - 'm<=cap'     = Genuine misses: model ingested the frames but failed to trigger.")
    print("  - '*_w' metrics= Recall/Precision/F1 calculated ONLY over triggers and predictions <= cap.")
    print("  If the 300-600s drop disappears in '*_w', the decay is from the coverage cap, not attention dilution.")
    print("=" * 88)


if __name__ == "__main__":
    main()
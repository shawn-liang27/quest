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
import json, argparse
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
    ap.add_argument("--gt_file", required=True)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max_frames", type=int, default=400)
    ap.add_argument("--tolerance", type=float, default=5.0)
    args = ap.parse_args()

    cap_sec = args.max_frames / args.fps
    preds = load_pred(args.pred_file)
    gt = load_gt(args.gt_file)

    # ---- duration distribution of the visual-only subset ----
    durs = sorted(v["duration"] for v in gt.values() if v["duration"])
    print("=" * 70)
    print("VISUAL-ONLY SUBSET — duration distribution")
    print("=" * 70)
    if durs:
        n = len(durs)
        print(f"  videos: {n}")
        print(f"  min/median/max duration: "
              f"{durs[0]:.0f}s / {durs[n//2]:.0f}s / {durs[-1]:.0f}s")
        print(f"  videos longer than cap ({cap_sec:.0f}s): "
              f"{sum(1 for d in durs if d > cap_sec)}")
        for thr in (180, 300, 400, 600):
            print(f"  videos > {thr}s: {sum(1 for d in durs if d > thr)}")

    # ---- per-bucket: triggers, and how many are beyond the coverage cap ----
    print("\n" + "=" * 70)
    print(f"COVERAGE CAP = {cap_sec:.0f}s  (fps={args.fps}, max_frames={args.max_frames})")
    print("=" * 70)

    # only consider samples actually run
    run = {qid: g for qid, g in gt.items() if qid in preds}
    print(f"  scored samples (in pred file): {len(run)}\n")

    per_bucket = defaultdict(lambda: {"trigs": 0, "beyond_cap": 0,
                                      "detected": 0, "missed_beyond": 0,
                                      "missed_within": 0})
    for qid, g in run.items():
        fires = preds[qid]
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

    print(f"  {'bucket':12s} {'trigs':>6s} {'beyond_cap':>11s} "
          f"{'recall':>7s} {'miss>cap':>9s} {'miss<=cap':>10s} {'recall_within':>14s}")
    for lo, hi in BUCKETS:
        b = f"{lo}-{hi if hi < 1e9 else 'inf'}s"
        pb = per_bucket.get(b)
        if not pb or pb["trigs"] == 0:
            continue
        recall = pb["detected"] / pb["trigs"]
        within = pb["trigs"] - pb["beyond_cap"]
        # recall computed ONLY over triggers the model could actually see
        det_within = pb["detected"]   # detected are by definition within reach
        recall_within = det_within / within if within > 0 else float("nan")
        print(f"  {b:12s} {pb['trigs']:6d} {pb['beyond_cap']:11d} "
              f"{recall:7.3f} {pb['missed_beyond']:9d} {pb['missed_within']:10d} "
              f"{recall_within:14.3f}")

    print("\nInterpretation:")
    print("  - 'beyond_cap' = triggers the model NEVER ingested (truncation).")
    print("  - 'miss>cap'   = misses explained by truncation (not decay).")
    print("  - 'miss<=cap'  = genuine misses: model saw the frames, didn't fire.")
    print("  - 'recall_within' = recall over only the triggers the model could see.")
    print("  If the 300-600s recall drop disappears in recall_within, the decay")
    print("  was the coverage cap, NOT attention dilution.")
    print("=" * 70)


if __name__ == "__main__":
    main()
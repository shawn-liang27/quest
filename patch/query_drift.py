"""
Within-video query-relevance drift test.

Hypothesis: query conditioning of FIRING is not sustained — fires get progressively
LESS query-relevant as the stream goes on (the model anchors early, then drifts into
generic scene captioning).

For each video we have its fires in time order, each with sim_query (cos of fire text
vs query). We test whether sim_query DECLINES with fire order/time, per video, then
aggregate the per-video slopes.

Two orderings tested:
  - by fire INDEX (1st fire, 2nd, ...): tests "conditioning decays per successive fire"
  - by fire TIME  (seconds):            tests "conditioning decays over video time"

Outputs:
  - mean/median per-video slope (negative => relevance declines => decay supported)
  - sign test: fraction of videos with negative slope (vs 50% null)
  - first-vs-last fire relevance (paired): is the first fire more query-relevant than
    the last, within the same video?
  - optional split by TP/FP and by task

A robust negative slope across videos turns the 9-sample "drift" reading into a measured
finding. A slope ~0 means firing relevance does NOT decay — the drift was anecdotal.

Input: a fires file with per-fire sim_query + qid + time. Easiest source is the
fp_samples.jsonl-style records, but for full coverage re-run the content scorer with
--sample_n large, OR point this at a jsonl where each line has:
    {qid, time, sim_query, is_tp(optional), task(optional)}

Usage:
  python diag_query_drift.py --fires fires_with_simquery.jsonl
  python diag_query_drift.py --fires fp_samples.jsonl --min_fires 4
"""
import json, argparse
from collections import defaultdict
import numpy as np


def load_fires(path):
    by_vid = defaultdict(list)
    for line in open(path):
        if not line.strip():
            continue
        d = json.loads(line)
        qid = d["qid"]
        if "sim_query" not in d or "time" not in d:
            continue
        by_vid[qid].append({
            "time": float(d["time"]),
            "sim_query": float(d["sim_query"]),
            "is_tp": d.get("is_tp", None),
            "task": d.get("task", "?"),
        })
    # sort each video's fires by time
    for q in by_vid:
        by_vid[q].sort(key=lambda r: r["time"])
    return by_vid


def slope(xs, ys):
    """Least-squares slope of ys vs xs; None if <2 points or no x-variance."""
    if len(xs) < 2:
        return None
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if xs.std() == 0:
        return None
    return float(np.polyfit(xs, ys, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fires", required=True,
                    help="jsonl with per-fire {qid, time, sim_query, [is_tp], [task]}")
    ap.add_argument("--min_fires", type=int, default=4,
                    help="only use videos with at least this many fires")
    ap.add_argument("--by", choices=["index", "time", "both"], default="both")
    args = ap.parse_args()

    by_vid = load_fires(args.fires)
    vids = {q: f for q, f in by_vid.items() if len(f) >= args.min_fires}
    print("=" * 64)
    print(f"QUERY-RELEVANCE DRIFT  (videos with >={args.min_fires} fires: {len(vids)})")
    print("=" * 64)
    if not vids:
        raise SystemExit("no videos meet min_fires; lower --min_fires or supply more fires")

    def report(order_key, label):
        slopes, first_last = [], []
        for q, fires in vids.items():
            sq = [f["sim_query"] for f in fires]
            if order_key == "index":
                xs = list(range(len(fires)))
            else:  # time
                xs = [f["time"] for f in fires]
            s = slope(xs, sq)
            if s is not None:
                slopes.append(s)
            first_last.append((sq[0], sq[-1]))
        slopes = np.array(slopes)
        firsts = np.array([a for a, _ in first_last])
        lasts = np.array([b for _, b in first_last])
        neg_frac = float((slopes < 0).mean())
        print(f"\n[{label}]  n_videos={len(slopes)}")
        print(f"  mean slope   : {slopes.mean():+.5f}   median: {np.median(slopes):+.5f}")
        print(f"  videos with NEGATIVE slope (relevance declines): {neg_frac:.1%}")
        print(f"    (50% = no drift; >>50% = relevance declines over {label})")
        print(f"  first-fire sim_query (mean): {firsts.mean():.3f}")
        print(f"  last-fire  sim_query (mean): {lasts.mean():.3f}")
        print(f"  paired first>last in {(firsts > lasts).mean():.1%} of videos "
              f"(mean drop {(firsts - lasts).mean():+.3f})")
        # simple sign-test p-ish: how far from 50/50
        from math import comb
        n = len(slopes); k = int((slopes < 0).sum())
        # two-sided binomial tail (small n ok)
        p = sum(comb(n, i) for i in range(min(k, n - k) + 1)) * 2 / (2 ** n) if n <= 1000 else None
        if p is not None:
            print(f"  sign-test p (slope!=0): {min(p,1.0):.4f}")

    if args.by in ("index", "both"):
        report("index", "by fire index")
    if args.by in ("time", "both"):
        report("time", "by video time")

    # ---- TP vs FP split, if labels present ----
    has_tp = any(f.get("is_tp") is not None for fires in vids.values() for f in fires)
    if has_tp:
        print("\n--- relevance by fire position, TP vs FP ---")
        for lab, want in [("TP", True), ("FP", False)]:
            vals = [f["sim_query"] for fires in vids.values() for f in fires
                    if f["is_tp"] == want]
            if vals:
                print(f"  {lab}: n={len(vals)} mean sim_query={np.mean(vals):.3f}")

    print("\nInterpretation:")
    print("  negative mean slope + >>50% videos declining + first>last")
    print("    => query relevance of firing DECAYS across the stream (measured)")
    print("  slope ~ 0 and ~50% => no sustained-conditioning decay; drift was anecdotal")
    print("=" * 64)


if __name__ == "__main__":
    main()
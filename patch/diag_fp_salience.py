"""
Step-1 diagnostic (Causes B and C): is the over-firing driven by visual
salience / scene-change, independent of the task?

For every prediction it classifies each fire as TP (within tol of a GT trigger)
or FP (false alarm), then computes two visual signals from the video at:
  - FP timestamps      (where the model wrongly fired)
  - TP/GT timestamps    (where it should fire)
  - random timestamps   (baseline rate)

Signals per timestamp:
  - busyness : mean absolute frame-to-frame pixel difference in a small window
               (Cause B: do FPs land on visually busy moments?)
  - change   : histogram-difference spike between adjacent frames
               (Cause C: do FPs land on scene changes, or persist in static scenes?)

Outputs distributions + a simple comparison. If FP busyness/change >> random
and is task-independent, the over-firing is salience/change-driven, not
task-conditioned -> "add a context token" is NOT the indicated fix.

Usage:
  python diag_fp_salience.py \
      --pred_file outputs/full_2/omnipro_visual-pred.jsonl \
      --gt_file   /mnt/data0/sgl57/data/omnipro/metadata.jsonl \
      --video_dir /mnt/data0/sgl57/data/omnipro/raw_videos \
      --tolerance 3.0 --max_videos 60
"""
import json, argparse, os, random
from collections import defaultdict
import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("needs opencv: pip install opencv-python-headless")


def load_pred(path):
    preds = {}
    for line in open(path):
        if not line.strip():
            continue
        d = json.loads(line)
        qid = d.get("question_id") or d.get("id")
        preds[qid] = [float(r["time"]) for r in d.get("model_response_list", [])
                      if r.get("role") == "assistant" and "time" in r]
    return preds


def load_gt(path):
    gt = {}
    for line in open(path):
        if not line.strip():
            continue
        row = json.loads(line)
        raw = row["ground_truth"]
        trigs = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(trigs, dict):
            trigs = trigs.get("triggers", [])
        times = []
        for t in trigs:
            if "trigger_time_sec" in t:
                times.append(float(t["trigger_time_sec"]))
        gt[row["id"]] = {
            "task": row["task"],
            "gt_times": times,
            "duration": row.get("duration"),
            "file_name": row.get("file_name"),
        }
    return gt


def classify_fires(fire_times, gt_times, tol):
    """Return (tp_times, fp_times). A fire is TP if within tol of any GT."""
    tp, fp = [], []
    for ft in fire_times:
        if any(abs(ft - g) <= tol for g in gt_times):
            tp.append(ft)
        else:
            fp.append(ft)
    return tp, fp


def sample_signals(video_path, times, win=0.5, fps_probe=None):
    """For each timestamp, compute (busyness, change) using frames around it.
    busyness = mean abs diff of two frames `win` apart.
    change   = correlation-distance of grayscale histograms of those frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = []
    for t in times:
        f1 = int(max(0, (t - win)) * fps)
        f2 = int((t + win) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f1)
        ok1, a = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, f2)
        ok2, b = cap.read()
        if not (ok1 and ok2):
            continue
        a = cv2.cvtColor(cv2.resize(a, (160, 90)), cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(cv2.resize(b, (160, 90)), cv2.COLOR_BGR2GRAY)
        busy = float(np.mean(np.abs(a.astype(int) - b.astype(int))))
        ha = cv2.calcHist([a], [0], None, [32], [0, 256]); ha /= (ha.sum() + 1e-9)
        hb = cv2.calcHist([b], [0], None, [32], [0, 256]); hb /= (hb.sum() + 1e-9)
        change = float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA))
        out.append((busy, change))
    cap.release()
    return out


def summarize(name, vals):
    if not vals:
        print(f"  {name:12s}: (no samples)"); return
    b = np.array([v[0] for v in vals]); c = np.array([v[1] for v in vals])
    print(f"  {name:12s}: n={len(vals):5d}  "
          f"busyness mean={b.mean():6.2f} med={np.median(b):6.2f}   "
          f"change mean={c.mean():.3f} med={np.median(c):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True)
    ap.add_argument("--gt_file", default="/mnt/data0/sgl57/data/omnipro/metadata.jsonl")
    ap.add_argument("--video_dir", default="/mnt/data0/sgl57/data/omnipro/raw_videos")
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--max_videos", type=int, default=60,
                    help="cap videos processed (frame reads are slow)")
    ap.add_argument("--per_task", action="store_true",
                    help="also break the FP signal down by task")
    args = ap.parse_args()

    preds = load_pred(args.pred_file)
    gt = load_gt(args.gt_file)
    random.seed(0)

    fp_all, tp_all, rand_all = [], [], []
    fp_by_task = defaultdict(list)
    n_done = 0

    for qid, info in gt.items():
        if qid not in preds:
            continue
        if n_done >= args.max_videos:
            break
        fname = info["file_name"] or ""
        vpath = os.path.join(args.video_dir, os.path.basename(fname))
        if not os.path.exists(vpath):
            # try the file_name as-is relative to video_dir parent
            alt = os.path.join(os.path.dirname(args.video_dir.rstrip("/")), fname)
            vpath = vpath if os.path.exists(vpath) else alt
            if not os.path.exists(vpath):
                continue

        tp_t, fp_t = classify_fires(preds[qid], info["gt_times"], args.tolerance)
        dur = info["duration"] or 0
        rand_t = [random.uniform(0, dur) for _ in range(max(len(fp_t), 3))] if dur else []

        fp_sig = sample_signals(vpath, fp_t)
        tp_sig = sample_signals(vpath, info["gt_times"])
        rd_sig = sample_signals(vpath, rand_t)

        fp_all += fp_sig; tp_all += tp_sig; rand_all += rd_sig
        fp_by_task[info["task"]] += fp_sig
        n_done += 1

    print("=" * 68)
    print(f"FP-salience diagnostic  (videos processed={n_done}, tol={args.tolerance}s)")
    print("=" * 68)
    print("\nVisual signal at different timestamp types:")
    summarize("FALSE ALARM", fp_all)
    summarize("TRUE EVENT",  tp_all)
    summarize("RANDOM",      rand_all)

    print("\nInterpretation:")
    if fp_all and rand_all:
        fb = np.mean([v[0] for v in fp_all]); rb = np.mean([v[0] for v in rand_all])
        print(f"  FP busyness vs random: {fb:.2f} vs {rb:.2f} "
              f"({'FP at busier moments -> salience-driven (Cause B)' if fb > rb*1.15 else 'FP not notably busier -> NOT pure salience'})")

    if args.per_task:
        print("\nFP busyness by task (task-independence check for Cause B):")
        for task, vals in fp_by_task.items():
            if vals:
                b = np.mean([v[0] for v in vals])
                print(f"  {task:30s} n={len(vals):4d}  busyness={b:6.2f}")
        print("  (similar busyness across tasks => over-firing is task-independent)")
    print("=" * 68)


if __name__ == "__main__":
    main()
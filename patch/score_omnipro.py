import json, argparse
from collections import defaultdict

# OmniPro official library (must be importable: run from their proactive_eval dir
# or add it to PYTHONPATH).
from scorer import evaluate_sample, aggregate

ALERT_TASKS = {"instant_event_alert", "semantic_condition_alert"}


def load_pred(path):
    """qid -> list of {t_sec, raw} (their expected emit shape)."""
    preds = {}
    for line in open(path):
        if not line.strip():
            continue
        d = json.loads(line)
        qid = d.get("question_id") or d.get("id")
        emits = []
        for r in d.get("model_response_list", []):
            if r.get("role") != "assistant":
                continue
            if "time" not in r:
                continue
            emits.append({"t_sec": float(r["time"]),
                          "raw": r.get("content", "")})
        preds[qid] = emits
    return preds


def load_gt(path, tasks=None, visual_only=True):
    """id -> {task, triggers(list of GT dicts), duration}.

    Reads REAL OmniPro metadata: ground_truth is a JSON string holding a
    bare list of trigger dicts. Optionally filter to `tasks` and visual-only.
    """
    gt = {}
    for line in open(path):
        if not line.strip():
            continue
        row = json.loads(line)
        if visual_only and row.get("audio_dependency") != "none":
            continue
        if tasks and row["task"] not in tasks:
            continue
        raw = row["ground_truth"]
        trigs = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(trigs, dict):          # tolerate wrapped, just in case
            trigs = trigs.get("triggers", [])
        gt[row["id"]] = {
            "task": row["task"],
            "triggers": trigs,
            "duration": row.get("duration"),
            "is_null": len(trigs) == 0,
        }
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True)
    ap.add_argument("--gt_file", default="/mnt/data0/sgl57/data/omnipro/metadata.jsonl")
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="OmniPro tasks to score (default: ALL tasks). "
                         "Pass specific task names to restrict.")
    ap.add_argument("--all_audio", action="store_true",
                    help="include audio-dependent samples (default: visual-only)")
    args = ap.parse_args()

    preds = load_pred(args.pred_file)
    gt = load_gt(args.gt_file, tasks=set(args.tasks) if args.tasks else None,
                 visual_only=not args.all_audio)

    per_sample = []          # for OmniPro aggregate() — alert tasks only
    null_rows = []
    extra = {}               # qid -> over-firing metrics

    for qid, info in gt.items():
        if qid not in preds:
            continue          # not yet run (partial run) — don't count as a miss
        emits = preds.get(qid, [])

        if info["is_null"]:
            null_rows.append({"id": qid, "fires": len(emits),
                              "duration": info["duration"]})
            continue

        sample_pred = {
            "id": qid,
            "task": info["task"],
            "predictions": emits,
            "ground_truth": info["triggers"],
        }
        res = evaluate_sample(sample_pred, tolerance=args.tolerance)
        per_sample.append(res)

        # over-firing metrics their lib omits
        dur_min = (info["duration"] or 0) / 60.0
        fa_per_min = res["fp"] / dur_min if dur_min > 0 else None
        dts = [m["dt"] for m in res["per_match"]]
        med_dt = sorted(dts)[len(dts)//2] if dts else None
        extra[qid] = {"task": info["task"],
                      "fa_per_min": fa_per_min, "median_dt": med_dt,
                      "n_emits": res["num_emits"], "n_gt": res["num_gt"]}

    # ---- official aggregation (temporal P/R/F1, content is no-op for alerts) ----
    agg = aggregate(per_sample) if per_sample else {}

    print("=" * 72)
    print(f"OmniPro firing metrics — ALL tasks (tol={args.tolerance}s, "
          f"visual_only={not args.all_audio})")
    print("=" * 72)
    task_keys = [k for k in agg if k != "overall"] + (["overall"] if "overall" in agg else [])
    for task in task_keys:
        m = agg[task]
        ca = m.get("content_accuracy")
        ca_str = f"{ca:.3f}" if ca is not None else "n/a"
        print(f"\n[{task}]  n={m['n_samples']}")
        print(f"  TIME    P={m['time_precision']:.3f}  R={m['time_recall']:.3f}  "
              f"F1={m['time_f1']:.3f}   (TP={m['tp_time']} FP={m['fp']} FN={m['fn']})")
        print(f"  CONTENT acc={ca_str}   JOINT F1={m['joint_f1']:.3f}")

    # ---- over-firing summary, per task ----
    if extra:
        by_task = defaultdict(list)
        for v in extra.values():
            by_task[v["task"]].append(v)
        print("\n" + "-" * 72)
        print("OVER-FIRING (per task)")
        for task, vs in by_task.items():
            fas = [v["fa_per_min"] for v in vs if v["fa_per_min"] is not None]
            dts = [v["median_dt"] for v in vs if v["median_dt"] is not None]
            ec = sorted(v["n_emits"] for v in vs)
            mean_fa = f"{sum(fas)/len(fas):.2f}" if fas else "n/a"
            med_dt = f"{sorted(dts)[len(dts)//2]:.2f}s" if dts else "n/a"
            print(f"  [{task}] FA/min={mean_fa}  fire-time-err={med_dt}  "
                  f"emits min/med/max={ec[0]}/{ec[len(ec)//2]}/{ec[-1]}")

    # ---- nulls ----
    if null_rows:
        print("\n[null queries — any fire = false conditioning]")
        n_fire = sum(1 for r in null_rows if r["fires"] > 0)
        print(f"  {n_fire}/{len(null_rows)} null queries fired")
        for r in null_rows:
            verdict = "PASS" if r["fires"] == 0 else f"FAIL ({r['fires']})"
            print(f"    {r['id'].split('::')[-1]:20s} {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
"""
False-alarm content analysis (semantic encoder, cosine).

Each false alarm is described along THREE INDEPENDENT axes (not one mixed bucket):

  1. CONTENT vs ground truth (what the fire says vs the nearest GT response):
       - MATCHES_GT : sim_resp >= gt_match_sim   (describes the real event correctly)
       - UNRELATED  : sim_resp <  gt_match_sim   ("NOVEL" content — nothing to do with GT)

  2. SELF-REPETITION (does the model repeat ITS OWN earlier output in this video?):
       - REPEAT     : sim to some earlier fire of the same video >= repeat_sim
       - FIRST_TIME : not a repeat of anything said earlier
     (This is the real "redundant" measure: model keeps saying the same thing.)

  3. TIME vs nearest GT trigger:
       - NEAR : |dist| <= near_window
       - FAR  : otherwise

  Plus query relevance (sim_query), reported relative to the TP distribution.

These are independent — e.g. a fire can be UNRELATED + REPEAT + FAR, or MATCHES_GT +
FIRST_TIME + NEAR (a near-hit just outside tolerance). The previous single "category"
mislabeled near-hits as "REDUNDANT" and conflated time-distance with content; this
version keeps them separate.

Usage:
  python fp_content.py \
      --pred_file outputs/.../omnipro_visual-pred.jsonl \
      --gt_file   /mnt/data0/sgl57/data/omnipro/metadata.jsonl \
      --tolerance 3.0 --near_window 10.0 \
      --encoder sentence-transformers/all-mpnet-base-v2 \
      --gt_match_sim 0.5 --repeat_sim 0.8 \
      --sample_out fp_samples.jsonl --sample_n 100
"""
import json, argparse
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer


def load_pred(path):
    preds = {}
    for line in open(path):
        if not line.strip():
            continue
        d = json.loads(line)
        qid = d.get("question_id") or d.get("id")
        fires, query = [], None
        for r in d.get("model_response_list", []):
            if r.get("role") == "user" and query is None:
                query = r.get("content", "")
            if r.get("role") == "assistant" and "time" in r:
                fires.append({"time": float(r["time"]), "text": r.get("content", "")})
        preds[qid] = {"fires": fires, "query": query}
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
        gt[row["id"]] = {
            "task": row["task"],
            "question": row.get("question", ""),
            "triggers": [{"time": float(t["trigger_time_sec"]),
                          "response": t.get("response", "")}
                         for t in trigs if "trigger_time_sec" in t],
        }
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True)
    ap.add_argument("--gt_file", required=True)
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--near_window", type=float, default=10.0)
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--gt_match_sim", type=float, default=0.5,
                    help="cos(fire, GT_response) >= this => fire describes the real event")
    ap.add_argument("--repeat_sim", type=float, default=0.8,
                    help="cos(fire, an earlier fire in same video) >= this => self-repeat")
    ap.add_argument("--sample_out", default="fp_samples.jsonl")
    ap.add_argument("--sample_n", type=int, default=100)
    args = ap.parse_args()

    enc = SentenceTransformer(args.encoder)
    preds = load_pred(args.pred_file)
    gt = load_gt(args.gt_file)

    # ---- collect fires, keep per-video order for self-repetition ----
    fire_recs = []
    for qid, info in gt.items():
        if qid not in preds:
            continue
        trigs = info["triggers"]
        gt_times = [t["time"] for t in trigs]
        query = info["question"] or preds[qid]["query"] or ""
        fires_sorted = sorted(preds[qid]["fires"], key=lambda f: f["time"])
        for idx, fire in enumerate(fires_sorted):
            ft = fire["time"]
            is_tp = any(abs(ft - g) <= args.tolerance for g in gt_times)
            nearest = min(trigs, key=lambda t: abs(ft - t["time"])) if trigs else None
            dist = abs(ft - nearest["time"]) if nearest else float("inf")
            fire_recs.append({
                "qid": qid, "task": info["task"], "time": ft, "text": fire["text"],
                "is_tp": is_tp, "dist": dist,
                "nearest_resp": nearest["response"] if nearest else "",
                "query": query,
                "vid_fire_idx": idx,   # position among this video's fires
            })

    if not fire_recs:
        raise SystemExit("no fires found / no overlap with gt")

    # ---- batch-encode ----
    def emb(strings):
        return enc.encode(strings, normalize_embeddings=True, batch_size=256,
                          show_progress_bar=True)
    E_text = emb([r["text"] for r in fire_recs])
    E_resp = emb([r["nearest_resp"] for r in fire_recs])
    E_query = emb([r["query"] for r in fire_recs])
    for i, r in enumerate(fire_recs):
        r["_e"] = E_text[i]
        r["sim_resp"] = float((E_text[i] * E_resp[i]).sum())
        r["sim_query"] = float((E_text[i] * E_query[i]).sum())

    # ---- self-repetition: max cos to any EARLIER fire in the same video ----
    by_vid = defaultdict(list)
    for r in fire_recs:
        by_vid[r["qid"]].append(r)
    for qid, recs in by_vid.items():
        recs.sort(key=lambda r: r["time"])
        for i, r in enumerate(recs):
            if i == 0:
                r["max_sim_prev"] = 0.0
            else:
                prev = np.stack([recs[j]["_e"] for j in range(i)])
                r["max_sim_prev"] = float((prev @ r["_e"]).max())
    for r in fire_recs:
        del r["_e"]

    tp = [r for r in fire_recs if r["is_tp"]]
    fp = [r for r in fire_recs if not r["is_tp"]]
    n_fp = len(fp)

    # ---- classify FPs on the three independent axes ----
    content = defaultdict(int)     # MATCHES_GT vs UNRELATED(NOVEL)
    repetition = defaultdict(int)  # REPEAT vs FIRST_TIME
    timing = defaultdict(int)      # NEAR vs FAR
    by_task = defaultdict(lambda: defaultdict(int))
    samples = []
    for r in fp:
        c = "MATCHES_GT" if r["sim_resp"] >= args.gt_match_sim else "UNRELATED(novel content)"
        rep = "REPEAT(of own earlier fire)" if r["max_sim_prev"] >= args.repeat_sim else "FIRST_TIME"
        tm = "NEAR" if r["dist"] <= args.near_window else "FAR"
        content[c] += 1; repetition[rep] += 1; timing[tm] += 1
        by_task[r["task"]][c] += 1
        r["content_cls"] = c; r["repeat_cls"] = rep; r["time_cls"] = tm
        if len(samples) < args.sample_n:
            samples.append({k: (round(r[k], 3) if isinstance(r[k], float) else r[k])
                            for k in ("qid","task","time","text","nearest_resp","query",
                                      "dist","sim_resp","sim_query","max_sim_prev",
                                      "content_cls","repeat_cls","time_cls")})

    fp_q = np.array([r["sim_query"] for r in fp])
    tp_q = np.array([r["sim_query"] for r in tp]) if tp else np.array([0.0])
    tp_q_med = float(np.median(tp_q))
    tp_resp_med = float(np.median([r["sim_resp"] for r in tp])) if tp else 0.0

    print("=" * 72)
    print(f"FALSE-ALARM CONTENT (encoder={args.encoder})")
    print(f"tol={args.tolerance}s near={args.near_window}s "
          f"gt_match>={args.gt_match_sim} repeat>={args.repeat_sim}")
    print("=" * 72)
    print(f"TP fires: {len(tp)}   FP fires: {n_fp}")
    print(f"(calibration: TP sim_resp median={tp_resp_med:.3f}, "
          f"TP sim_query median={tp_q_med:.3f})\n")

    print("AXIS 1 — CONTENT vs ground-truth response:")
    for c, n in sorted(content.items(), key=lambda x: -x[1]):
        print(f"   {c:30s} {n:6d} ({n/n_fp:.1%})")
    print("   (UNRELATED = the fire says something unrelated to the real event)")

    print("\nAXIS 2 — SELF-REPETITION (model repeats its OWN earlier output):")
    for c, n in sorted(repetition.items(), key=lambda x: -x[1]):
        print(f"   {c:30s} {n:6d} ({n/n_fp:.1%})")
    print("   (REPEAT = this fire restates something the model already said earlier)")

    print("\nAXIS 3 — TIMING vs nearest true trigger:")
    for c, n in sorted(timing.items(), key=lambda x: -x[1]):
        print(f"   {c:30s} {n:6d} ({n/n_fp:.1%})")
    print("   (NEAR but counted FP = near-hit just outside tolerance)")

    print("\n--- query relevance (fire vs query), relative to TP ---")
    print(f"  TP median sim_query: {tp_q_med:.3f}   FP median sim_query: {np.median(fp_q):.3f}")
    print(f"  fraction of FP less query-relevant than median TP: {float((fp_q<tp_q_med).mean()):.1%}")

    print("\n--- per-task UNRELATED-content share ---")
    for task, b in by_task.items():
        tot = sum(b.values())
        un = b.get("UNRELATED(novel content)", 0)
        print(f"  {task:28s} n={tot:5d}  unrelated={un/tot:.0%}")

    with open(args.sample_out, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"\n  wrote {len(samples)} samples (3 axis labels each) -> {args.sample_out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
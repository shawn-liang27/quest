"""
False-alarm content analysis with a SEMANTIC text encoder (cosine), not lexical ROUGE-L.

Two semantic signals per fire (sentence-transformers, MiniLM by default):
  1. cos(fire_text, nearest_GT_response)  -> redundancy / near-miss classification
  2. cos(fire_text, query)                -> is the fire even ABOUT what was asked?

The key test: compare NOVEL false-alarm relevance-to-query against TRUE-POSITIVE
relevance-to-query. If novel FPs are as query-related as TPs, firing isn't gating on
the query at all (indiscriminate). TPs serve as the calibration reference, so we don't
rely on an absolute cosine threshold across registers (instruction vs description).

Thresholds are calibrated, not guessed:
  - sim-to-GT redundancy threshold: default 0.6 (MiniLM near-paraphrase regime),
    overridable; we also print the TP-vs-GT cosine distribution so you can sanity-set it.
  - query-relevance: judged RELATIVE to the TP distribution (median TP cosine), not an
    absolute cutoff.

Usage:
  python diag_fp_content.py \
      --pred_file outputs/.../omnipro_visual-pred.jsonl \
      --gt_file   /mnt/data0/sgl57/data/omnipro/metadata.jsonl \
      --tolerance 3.0 --near_window 10.0 \
      --encoder sentence-transformers/all-MiniLM-L6-v2 \
      --redundant_sim 0.6 --sample_out fp_samples.jsonl --sample_n 100
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
    ap.add_argument("--redundant_sim", type=float, default=0.6,
                    help="cos(fire, GT_response) above this = restates the GT event")
    ap.add_argument("--sample_out", default="fp_samples.jsonl")
    ap.add_argument("--sample_n", type=int, default=100)
    args = ap.parse_args()

    enc = SentenceTransformer(args.encoder)

    preds = load_pred(args.pred_file)
    gt = load_gt(args.gt_file)

    # ---- collect all texts, encode once in batch (fast) ----
    fire_recs = []   # (qid, task, time, text, is_tp, nearest_resp, query)
    for qid, info in gt.items():
        if qid not in preds:
            continue
        trigs = info["triggers"]
        gt_times = [t["time"] for t in trigs]
        query = info["question"] or preds[qid]["query"] or ""
        for fire in preds[qid]["fires"]:
            ft = fire["time"]
            is_tp = any(abs(ft - g) <= args.tolerance for g in gt_times)
            nearest = min(trigs, key=lambda t: abs(ft - t["time"])) if trigs else None
            dist = abs(ft - nearest["time"]) if nearest else float("inf")
            fire_recs.append({
                "qid": qid, "task": info["task"], "time": ft, "text": fire["text"],
                "is_tp": is_tp, "dist": dist,
                "nearest_resp": nearest["response"] if nearest else "",
                "query": query,
            })

    if not fire_recs:
        raise SystemExit("no fires found / no overlap with gt")

    # batch-encode unique strings
    texts = [r["text"] for r in fire_recs]
    resps = [r["nearest_resp"] for r in fire_recs]
    queries = [r["query"] for r in fire_recs]
    E_text = enc.encode(texts, normalize_embeddings=True, batch_size=256, show_progress_bar=True)
    E_resp = enc.encode(resps, normalize_embeddings=True, batch_size=256, show_progress_bar=True)
    E_query = enc.encode(queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True)
    sim_resp = (E_text * E_resp).sum(1)     # cos, since normalized
    sim_query = (E_text * E_query).sum(1)

    for i, r in enumerate(fire_recs):
        r["sim_resp"] = float(sim_resp[i])
        r["sim_query"] = float(sim_query[i])

    tp = [r for r in fire_recs if r["is_tp"]]
    fp = [r for r in fire_recs if not r["is_tp"]]
    n_fp = len(fp)

    # ---- calibration reference: TP relevance-to-query distribution ----
    tp_q = np.array([r["sim_query"] for r in tp]) if tp else np.array([0.0])
    tp_q_med = float(np.median(tp_q))
    tp_resp = np.array([r["sim_resp"] for r in tp]) if tp else np.array([0.0])

    # ---- buckets for FPs ----
    buckets = defaultdict(int)
    by_task = defaultdict(lambda: defaultdict(int))
    fp_q = []
    samples = []
    for r in fp:
        near = r["dist"] <= args.near_window
        if near and r["sim_resp"] >= args.redundant_sim:
            cat = "REDUNDANT (near + semantically restates GT)"
        elif near:
            cat = "NEAR-MISS (near, different meaning)"
        else:
            cat = "NOVEL (far from any GT)"
        buckets[cat] += 1
        by_task[r["task"]][cat] += 1
        fp_q.append(r["sim_query"])
        if len(samples) < args.sample_n:
            samples.append({k: (round(r[k], 3) if isinstance(r[k], float) else r[k])
                            for k in ("qid","task","time","text","nearest_resp",
                                      "query","dist","sim_resp","sim_query")} | {"category": cat})

    fp_q = np.array(fp_q)

    print("=" * 70)
    print(f"FALSE-ALARM CONTENT (semantic, encoder={args.encoder})")
    print(f"tol={args.tolerance}s near={args.near_window}s redundant_sim>={args.redundant_sim}")
    print("=" * 70)
    print(f"TP fires: {len(tp)}   FP fires: {n_fp}\n")
    for cat, n in sorted(buckets.items(), key=lambda x: -x[1]):
        print(f"  {cat:44s} {n:6d} ({n/n_fp:.1%})")

    print("\n--- query relevance (cos of fire text vs query) ---")
    print(f"  TP fires : median={tp_q_med:.3f}  mean={tp_q.mean():.3f}")
    print(f"  FP fires : median={np.median(fp_q):.3f}  mean={fp_q.mean():.3f}")
    print(f"  GT-response match (TP): median={float(np.median(tp_resp)):.3f}")
    # the decisive comparison, judged RELATIVE to TP (no absolute threshold needed)
    frac_fp_below = float((fp_q < tp_q_med).mean())
    print(f"\n  fraction of FP fires LESS query-relevant than the median TP: {frac_fp_below:.1%}")
    print("  Read:")
    print("   - FP query-relevance ~ TP -> firing does NOT gate on the query")
    print("        (novel fires look as on-topic as correct ones = indiscriminate)")
    print("   - FP query-relevance << TP -> novel fires are off-topic; model has SOME")
    print("        query signal it fails to threshold (decision/representation gap)")

    print("\n--- per-task NOVEL share ---")
    for task, b in by_task.items():
        tot = sum(b.values())
        nov = b.get("NOVEL (far from any GT)", 0)
        print(f"  {task:28s} n={tot:5d}  novel={nov/tot:.0%}")

    with open(args.sample_out, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"\n  wrote {len(samples)} samples (with sim_resp, sim_query) -> {args.sample_out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
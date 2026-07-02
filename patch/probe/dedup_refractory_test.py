"""
Dedup + refractory post-filter, scored at MATCHED RECALL.

Hypothesis (untested until now): the majority FP class at deployment is self-repetition
(71-82% near-dup of a prior fire; 24% exact). The static separability probe is blind to
this: re-fires while an event-state persists sit on HIGH-score frames, so raising AUROC
would not remove them. A dedup/refractory rule deletes them BY CONSTRUCTION, independent
of the representation ceiling.

This applies two post-filters to the existing fire stream and re-scores:
  - REFRACTORY: after a kept fire, suppress any fire within `refractory` seconds.
  - DEDUP: suppress a fire whose text cos-sim to any kept fire within `dedup_window`
           seconds is >= `dedup_sim`.
Both only DELETE fires -> recall can only drop, precision can only rise. The question is
the trade: at matched recall vs baseline, does dedup give higher precision than simply
thresholding would? Since dedup removes a structurally different set (repeats) than
thresholding (low-score fires), it can beat the threshold curve.

Scoring uses the SAME 1-to-1 greedy trigger matching idea as your scorer (each GT hit
once, within tol). Precision/recall computed on the filtered fire list.

Usage:
  python dedup_refractory_test.py \
      --pred_file base-pred.jsonl --gt_file /mnt/data0/.../metadata.jsonl \
      --refractory 5 --dedup_sim 0.85 --dedup_window 30 --tol 3.0 \
      --encoder sentence-transformers/all-mpnet-base-v2
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


def load_fires(path):
    """qid -> list of {time, text} sorted by time."""
    out = {}
    for line in open(path):
        if not line.strip():
            continue
        d = json.loads(line)
        qid = d.get("question_id") or d.get("id")
        fires = [{"time": float(r["time"]), "text": r.get("content", "")}
                 for r in d.get("model_response_list", [])
                 if r.get("role") == "assistant" and "time" in r]
        out[qid] = sorted(fires, key=lambda f: f["time"])
    return out


def greedy_match(fire_times, gt_times, tol):
    """1-to-1 greedy: each GT matched to at most one fire within tol. Returns (TP, FP, FN)."""
    used_gt = [False] * len(gt_times)
    tp = 0
    for ft in sorted(fire_times):
        best, bestd = -1, tol + 1e-9
        for gi, g in enumerate(gt_times):
            if used_gt[gi]:
                continue
            dd = abs(ft - g)
            if dd <= tol and dd < bestd:
                best, bestd = gi, dd
        if best >= 0:
            used_gt[best] = True
            tp += 1
    fp = len(fire_times) - tp
    fn = len(gt_times) - sum(used_gt)
    return tp, fp, fn


def score(fires_by_qid, gt, tol):
    TP = FP = FN = 0
    for qid, gts in gt.items():
        fts = [f["time"] for f in fires_by_qid.get(qid, [])]
        tp, fp, fn = greedy_match(fts, gts, tol)
        TP += tp; FP += fp; FN += fn
    P = TP / max(TP + FP, 1)
    R = TP / max(TP + FN, 1)
    F1 = 0 if P + R == 0 else 2 * P * R / (P + R)
    return dict(P=P, R=R, F1=F1, TP=TP, FP=FP, FN=FN)


def apply_refractory(fires, refractory):
    if refractory <= 0:
        return fires
    kept = []
    last_t = -1e9
    for f in fires:               # already time-sorted
        if f["time"] - last_t >= refractory:
            kept.append(f); last_t = f["time"]
    return kept


def apply_dedup(fires, emb, dedup_sim, dedup_window):
    if dedup_sim >= 1.0 and dedup_window <= 0:
        return fires
    kept = []
    kept_e = []
    kept_t = []
    for i, f in enumerate(fires):
        e = emb[i]
        dup = False
        for ke, kt in zip(kept_e, kept_t):
            if f["time"] - kt <= dedup_window and float(e @ ke) >= dedup_sim:
                dup = True; break
        if not dup:
            kept.append(f); kept_e.append(e); kept_t.append(f["time"])
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True)
    ap.add_argument("--gt_file", required=True)
    ap.add_argument("--tol", type=float, default=3.0)
    ap.add_argument("--refractory", type=float, default=0.0, help="seconds; 0=off")
    ap.add_argument("--dedup_sim", type=float, default=1.01, help=">=1.01 = off")
    ap.add_argument("--dedup_window", type=float, default=30.0)
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    args = ap.parse_args()

    gt = load_gt(args.gt_file)
    fires = load_fires(args.pred_file)

    # baseline
    base = score(fires, gt, args.tol)

    # encode all fire texts once (only needed if dedup on)
    need_emb = args.dedup_sim < 1.0
    emb_by_qid = {}
    if need_emb:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(args.encoder)
        all_texts, idx = [], {}
        for qid, fs in fires.items():
            idx[qid] = (len(all_texts), len(all_texts) + len(fs))
            all_texts.extend([f["text"] for f in fs])
        E = enc.encode(all_texts, normalize_embeddings=True, batch_size=256,
                       show_progress_bar=True) if all_texts else np.zeros((0, 768))
        for qid, (a, b) in idx.items():
            emb_by_qid[qid] = E[a:b]

    # apply filters per video
    filtered = {}
    for qid, fs in fires.items():
        ff = apply_refractory(fs, args.refractory)
        if need_emb:
            # re-embed the refractory-surviving subset by index mapping
            keep_idx = [fs.index(x) for x in ff]
            e = emb_by_qid[qid][keep_idx] if len(keep_idx) else np.zeros((0, E.shape[1]))
            ff = apply_dedup(ff, e, args.dedup_sim, args.dedup_window)
        filtered[qid] = ff

    filt = score(filtered, gt, args.tol)

    print("=" * 60)
    print("DEDUP / REFRACTORY POST-FILTER")
    print(f"refractory={args.refractory}s  dedup_sim={args.dedup_sim}  "
          f"dedup_window={args.dedup_window}s  tol={args.tol}s")
    print("=" * 60)
    print(f"  {'':10s} {'P':>7s} {'R':>7s} {'F1':>7s} {'TP':>6s} {'FP':>7s}")
    print(f"  {'baseline':10s} {base['P']:7.3f} {base['R']:7.3f} {base['F1']:7.3f} "
          f"{base['TP']:6d} {base['FP']:7d}")
    print(f"  {'filtered':10s} {filt['P']:7.3f} {filt['R']:7.3f} {filt['F1']:7.3f} "
          f"{filt['TP']:6d} {filt['FP']:7d}")
    dFP = base['FP'] - filt['FP']
    dTP = base['TP'] - filt['TP']
    print(f"\n  removed {dFP} FP ({dFP/max(base['FP'],1):.1%} of FPs) and "
          f"{dTP} TP ({dTP/max(base['TP'],1):.1%} of TPs)")
    print(f"  precision {base['P']:.3f} -> {filt['P']:.3f}  (+{filt['P']-base['P']:.3f})")
    print(f"  recall    {base['R']:.3f} -> {filt['R']:.3f}  ({filt['R']-base['R']:+.3f})")
    print(f"  F1        {base['F1']:.3f} -> {filt['F1']:.3f}  ({filt['F1']-base['F1']:+.3f})")
    print("\n  KEY: dedup/refractory only DELETE fires. If it removes mostly FP and few TP")
    print("  -> the repeats were the FP class, and a large chunk of over-firing is")
    print("     decision-fixable (NOT representational). Compare this precision to what")
    print("     thresholding achieves at the SAME filtered recall (run matched_recall_test).")
    print("=" * 60)


if __name__ == "__main__":
    main()
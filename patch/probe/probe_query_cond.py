"""
Cheap probe test: does TRAINED query-conditioning extract more event-separability
from the (frozen) hidden state than h_t alone?

Re-prompting tested frozen, input-level query re-injection -> no lift. This tests the
DIFFERENT thing: a trained readout that conditions on the query at the representation
level. Three probes on the SAME baseline hidden states (layer 20 by default):

  1. h_t only            : baseline ceiling (~0.79)
  2. [h_t ; q]  concat   : linear/MLP probe with query embedding concatenated
  3. FiLM(h_t, q)        : query produces per-dim scale+shift on h_t, then linear head

If (2)/(3) beat (1) -> trained query-conditioning DOES add separability -> query
steering/FiLM worth building. If not -> query info isn't the missing signal -> build
onset/change features instead.

Uses BASELINE hstates (NOT the re-prompted npz — those already bake the query in).
Query embedding is joined per-video from OmniPro metadata and broadcast to frames.

Usage:
  python probe_query_cond.py \
      --npz /usr/homes/sgl57/quest/baseline/MMDuet/hstates/hstates.npz \
      --gt_file /mnt/data0/sgl57/data/omnipro/metadata.jsonl \
      --layer h_layer_20 --epochs 40
"""
import argparse, json
import numpy as np
import torch
import torch.nn as nn


def auroc(scores, labels):
    s = np.asarray(scores); y = np.asarray(labels)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s); ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = y.sum(); n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_queries(gt_file):
    """vid (question_id) -> query text. OmniPro id == question_id used by collector."""
    q = {}
    for line in open(gt_file):
        if not line.strip():
            continue
        r = json.loads(line)
        q[r["id"]] = r.get("question", "")
    return q


class ConcatProbe(nn.Module):
    def __init__(self, dh, dq, nonlinear=False, hid=512):
        super().__init__()
        d = dh + dq
        self.net = (nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, 2))
                    if nonlinear else nn.Linear(d, 2))
    def forward(self, h, q):
        return self.net(torch.cat([h, q], dim=-1))


class FiLMProbe(nn.Module):
    """Query produces per-dim gamma/beta on h_t, then a linear classifier."""
    def __init__(self, dh, dq, hid=256):
        super().__init__()
        self.to_gamma = nn.Sequential(nn.Linear(dq, hid), nn.GELU(), nn.Linear(hid, dh))
        self.to_beta = nn.Sequential(nn.Linear(dq, hid), nn.GELU(), nn.Linear(hid, dh))
        self.head = nn.Linear(dh, 2)
    def forward(self, h, q):
        return self.head(h * (1 + self.to_gamma(q)) + self.to_beta(q))


def train_eval(model, Xtr, qtr, ytr, Xva, qva, yva, device, epochs, pos_w, has_q=True):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w], device=device))
    Xt = torch.tensor(Xtr, device=device); yt = torch.tensor(ytr, device=device)
    qt = torch.tensor(qtr, device=device) if has_q else None
    bs = 4096
    for _ in range(epochs):
        model.train(); perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]; opt.zero_grad()
            out = model(Xt[idx], qt[idx]) if has_q else model(Xt[idx])
            lossf(out, yt[idx]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        Xv = torch.tensor(Xva, device=device)
        qv = torch.tensor(qva, device=device) if has_q else None
        out = model(Xv, qv) if has_q else model(Xv)
        sc = torch.softmax(out, dim=-1)[:, 1].cpu().numpy()
    return auroc(sc, yva)


class HOnly(nn.Module):
    def __init__(self, dh, nonlinear=False, hid=512):
        super().__init__()
        self.net = (nn.Sequential(nn.Linear(dh, hid), nn.GELU(), nn.Linear(hid, 2))
                    if nonlinear else nn.Linear(dh, 2))
    def forward(self, h):
        return self.net(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="BASELINE hstates npz (not re-prompted)")
    ap.add_argument("--gt_file", default="/mnt/data0/sgl57/data/omnipro/metadata.jsonl")
    ap.add_argument("--layer", default="h_layer_20")
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    y = d["y"].astype(np.int64); vid = d["vid"]
    H = d[args.layer].astype(np.float32)
    print(f"layer={args.layer}  frames={len(y)}  event_frac={y.mean():.3f}  videos={len(set(vid))}")

    # query embedding per video, broadcast to frames
    from sentence_transformers import SentenceTransformer
    qmap = load_queries(args.gt_file)
    uniq = sorted(set(vid))
    missing = [v for v in uniq if v not in qmap or not qmap[v]]
    if missing:
        print(f"WARNING: {len(missing)}/{len(uniq)} videos have no query in metadata "
              f"(e.g. {missing[:3]}); their query embedding will be zeros")
    enc = SentenceTransformer(args.encoder)
    uq_texts = [qmap.get(v, "") or "" for v in uniq]
    uq_emb = enc.encode(uq_texts, normalize_embeddings=True, batch_size=128, show_progress_bar=True)
    v2e = {v: uq_emb[i] for i, v in enumerate(uniq)}
    Q = np.stack([v2e[v] for v in vid]).astype(np.float32)
    dq = Q.shape[1]

    # grouped split by video
    rng = np.random.default_rng(args.seed)
    vids = np.array(uniq); rng.shuffle(vids)
    val = set(vids[:max(1, int(len(vids) * args.val_frac))])
    vm = np.array([v in val for v in vid])
    Xtr, qtr, ytr = H[~vm], Q[~vm], y[~vm]
    Xva, qva, yva = H[vm], Q[vm], y[vm]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_w = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    dh = H.shape[1]
    print(f"train {len(Xtr)} / val {len(Xva)}; dh={dh} dq={dq}; pos_w={pos_w:.2f}\n")

    print(f"  {'probe':28s} {'AUROC':>7s}")
    a = train_eval(HOnly(dh, False), Xtr, None, ytr, Xva, None, yva, device, args.epochs, pos_w, has_q=False)
    print(f"  {'h_t only (linear)':28s} {a:7.3f}")
    a = train_eval(HOnly(dh, True), Xtr, None, ytr, Xva, None, yva, device, args.epochs, pos_w, has_q=False)
    print(f"  {'h_t only (MLP)':28s} {a:7.3f}")
    a = train_eval(ConcatProbe(dh, dq, False), Xtr, qtr, ytr, Xva, qva, yva, device, args.epochs, pos_w)
    print(f"  {'[h_t;q] concat (linear)':28s} {a:7.3f}")
    a = train_eval(ConcatProbe(dh, dq, True), Xtr, qtr, ytr, Xva, qva, yva, device, args.epochs, pos_w)
    print(f"  {'[h_t;q] concat (MLP)':28s} {a:7.3f}")
    a = train_eval(FiLMProbe(dh, dq), Xtr, qtr, ytr, Xva, qva, yva, device, args.epochs, pos_w)
    print(f"  {'FiLM(h_t,q)':28s} {a:7.3f}")

    print("\n  concat/FiLM >> h_t only -> trained query-conditioning adds separability")
    print("        -> query steering/FiLM worth building")
    print("  concat/FiLM ~= h_t only -> query info is NOT the missing signal")
    print("        -> build onset/change features instead")


if __name__ == "__main__":
    main()
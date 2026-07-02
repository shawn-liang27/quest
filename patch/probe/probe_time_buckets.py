"""
Per-time-bucket separability of the firing representation.

Tests whether the probe's event/non-event AUROC on the hidden state CHANGES with
position in the video. Trains one probe (grouped-by-video split) on a chosen layer,
then evaluates AUROC separately within each time bucket of the val frames.

  - AUROC flat across buckets  -> representational weakness is STATIC (no temporal
                                   component; fix targets per-frame conditioning).
  - AUROC declines with time   -> the representation degrades at separating events
                                   later in the stream (long-context degradation;
                                   fix needs a temporal/re-anchoring component).

REQUIRES a per-frame timestamp. The collector must store a 'time' array in the npz.
If your current npz lacks it, add in _encode_frame's collector path:
    self.time.append(float(fr["time"]))         # in add_video, parallel to y/vid
and save it:  np.savez_compressed(..., time=np.asarray(self.time, dtype=np.float32))
Until then, pass --time_from_vid_order to approximate time by within-video fire order
(coarser, but still shows position effects).

Usage:
  python probe_time_buckets.py --npz hstates/hstates.npz --layer h_layer_20 --epochs 30
"""
import argparse
import numpy as np
import torch
import torch.nn as nn

BUCKETS = [(0,30),(30,60),(60,120),(120,180),(180,300),(300,600),(600,1e9)]


def auroc(scores, labels):
    s=np.asarray(scores); y=np.asarray(labels)
    if y.sum()==0 or y.sum()==len(y): return float("nan")
    order=np.argsort(s); ranks=np.empty_like(order,dtype=float); ranks[order]=np.arange(1,len(s)+1)
    n_pos=y.sum(); n_neg=len(y)-n_pos
    return (ranks[y==1].sum()-n_pos*(n_pos+1)/2)/(n_pos*n_neg)


class Probe(nn.Module):
    def __init__(self,d,nonlinear=False,h=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.Linear(h,2)) if nonlinear else nn.Linear(d,2)
    def forward(self,x): return self.net(x)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--npz",required=True)
    ap.add_argument("--layer",default=None,help="which h_layer_* to use; default=last")
    ap.add_argument("--epochs",type=int,default=30)
    ap.add_argument("--val_frac",type=float,default=0.3)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--nonlinear",action="store_true")
    ap.add_argument("--time_from_vid_order",action="store_true",
                    help="approximate time by within-video frame order if no 'time' array")
    args=ap.parse_args()

    d=np.load(args.npz,allow_pickle=True)
    y=d["y"].astype(np.int64); vid=d["vid"]
    layer_keys=sorted([k for k in d.files if k.startswith("h_layer_")],
                      key=lambda k:int(k.split("_")[-1]))
    layer=args.layer or layer_keys[-1]
    H=d[layer].astype(np.float32)
    print(f"layer={layer}  frames={len(y)}  event_frac={y.mean():.3f}  videos={len(set(vid))}")

    # timestamps
    if "time" in d.files:
        t=d["time"].astype(np.float32)
    elif args.time_from_vid_order:
        # approximate: rank within each video as pseudo-time
        t=np.zeros(len(y),dtype=np.float32)
        from collections import defaultdict
        idxs=defaultdict(list)
        for i,v in enumerate(vid): idxs[v].append(i)
        for v,ii in idxs.items():
            for rank,i in enumerate(ii): t[i]=rank   # 0,1,2,... per video
        print("WARNING: using within-video order as pseudo-time (no 'time' array in npz)")
    else:
        raise SystemExit("npz has no 'time' array. Re-dump with per-frame time, or pass "
                         "--time_from_vid_order for a coarse approximation.")

    # grouped split
    rng=np.random.default_rng(args.seed)
    vids=np.array(sorted(set(vid))); rng.shuffle(vids)
    val=set(vids[:max(1,int(len(vids)*args.val_frac))])
    vm=np.array([v in val for v in vid])
    Xtr,ytr=H[~vm],y[~vm]; Xva,yva,tva=H[vm],y[vm],t[vm]
    device="cuda" if torch.cuda.is_available() else "cpu"
    pos_w=float((ytr==0).sum()/max(1,(ytr==1).sum()))

    # train one probe
    model=Probe(H.shape[1],args.nonlinear).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    lossf=nn.CrossEntropyLoss(weight=torch.tensor([1.0,pos_w],device=device))
    Xt=torch.tensor(Xtr,device=device); yt=torch.tensor(ytr,device=device)
    bs=4096
    for _ in range(args.epochs):
        model.train(); perm=torch.randperm(len(Xt),device=device)
        for i in range(0,len(Xt),bs):
            idx=perm[i:i+bs]; opt.zero_grad()
            lossf(model(Xt[idx]),yt[idx]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        sc=torch.softmax(model(torch.tensor(Xva,device=device)),dim=-1)[:,1].cpu().numpy()

    print(f"\noverall val AUROC: {auroc(sc,yva):.3f}\n")
    print(f"  {'bucket':12s} {'n':>7s} {'events':>7s} {'AUROC':>7s}")
    use_order = ("time" not in d.files and args.time_from_vid_order)
    if use_order:
        # bucket by integer order ranges
        order_buckets=[(0,1),(1,3),(3,6),(6,10),(10,20),(20,1e9)]
        for lo,hi in order_buckets:
            m=(tva>=lo)&(tva<hi)
            if m.sum()==0: continue
            print(f"  order[{lo:.0f}-{hi if hi<1e9 else 'inf'}) {m.sum():7d} "
                  f"{int(yva[m].sum()):7d} {auroc(sc[m],yva[m]):7.3f}")
    else:
        for lo,hi in BUCKETS:
            m=(tva>=lo)&(tva<hi)
            if m.sum()==0: continue
            b=f"{lo}-{hi if hi<1e9 else 'inf'}s"
            print(f"  {b:12s} {m.sum():7d} {int(yva[m].sum()):7d} {auroc(sc[m],yva[m]):7.3f}")

    print("\n  flat across buckets -> static representational weakness")
    print("  declining           -> representation degrades over stream (long-context)")


if __name__=="__main__":
    main()
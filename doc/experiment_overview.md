# Proactive Firing — Experiment Log

All experiments on OmniPro visual-only (n=432, audio_dependency=none) unless noted.
Models: **MMDuet** (LLaVA-OneVision-Qwen2-7B, linear relevance/informative heads + threshold)
and **MMDuet2** (Qwen2.5-VL-3B, RL-trained, generative "reply vs NO REPLY" decision).

---

## 1. Baseline characterization

| # | Experiment | What it tested | Result | Takeaway |
|---|---|---|---|---|
| 1 | Baseline firing metrics | Precision/recall/F1 of firing | MMDuet P=0.093 R=0.863 F1=0.168 (14.5k FP); MMDuet2 P=0.13–0.16 R=0.62–0.65 F1=0.23–0.26 (~5k FP) | ~85–91% of all fires are false alarms. Over-firing is the universal failure across all 9 tasks and both models. |
| 2 | Fire-timing error on hits | Whether timing or selection is the bottleneck | Median fire-time error 1.0s (both models) | Timing on caught events is tight. Bottleneck is **whether** to fire, not **when**. |
| 3 | Emit-count distribution | Shape of over-firing | Bimodal: 0 to 222 emits/video (min/med/max 6/35/128 MMDuet) | Silent-or-spray; no usable operating point at the deployed threshold. |
| 4 | MMDuet vs MMDuet2 by F1 | Whether RL helps | MMDuet2 higher F1 despite lower recall; MMDuet sprays 3× more | RL traded recall for precision and won on F1, but precision still floor. Matches OmniPro leaderboard. |

---

## 2. Ruling out candidate causes (the diagnostic chain)

| # | Experiment | What it tested | Result | Takeaway |
|---|---|---|---|---|
| 5 | FP-vs-salience (`diag_fp_salience`) | Are FPs drawn to visually busy moments? | FP busyness 24.3 ≈ random 24.0; true events 2× more distinct (44.6) | **NOT salience-distraction.** Fires are indiscriminate, not lured by visual activity. |
| 6 | informative_head AUROC | Is the salience head useful? | AUROC 0.484 ≈ chance | Learned informativeness head carries no event signal; summing it dilutes the query signal. |
| 7 | Threshold sweep | Is it miscalibration? | No usable operating point (best P~0.35 @ R~0.1) | **NOT calibration.** No threshold on the existing score recovers precision. |
| 8 | Windowed score (`diag_windowed`) | Is it a temporal-integration gap? | mean/max/sum/ewma (k≤15) lift AUROC +0.003 | **NOT temporal integration.** KV cache already makes h_t temporally informed. |
| 9 | MLP vs linear probe | Is the signal nonlinearly hidden? | MLP ≈ linear at every layer (0.747 vs 0.748 final) | **NOT nonlinear readout.** Signal is linearly accessible; there just isn't more of it. |
| 10 | Coverage-cap / long-horizon (`diag_coverage_cap`) | Long-horizon decay? | Within-reach recall 0.81→0.60 over 0–300s; raw collapse was truncation artifact | **NOT long-horizon decay** (within testable range). Cache/coverage cap bounds it; OmniPro ≤595s too short to test true long-horizon. |

---

## 3. Representation probing (the core evidence)

| # | Experiment | What it tested | Result | Takeaway |
|---|---|---|---|---|
| 11 | Relevance-head AUROC | Deployed head's event-separability | 0.69 aggregate (0.53 alerts, 0.76 counting) | Deployed head is weak — but this measures the **head**, not the representation. |
| 12 | Fresh probe on frozen h_t (100 vid) | Representation's separability ceiling | Linear 0.748, MLP 0.747; best layer 20 = 0.79, final 28 = 0.75 | **Representation caps ~0.79**, peaks mid-network, degrades at final layer. Head undertrained (0.69→0.75 free). |
| 13 | Per-layer probe, full data (432 vid) | Confirm ceiling + peak at scale | Layer 20: 0.785/0.791; final 28: 0.774; lift +0.016 | Ceiling **~0.79 robust** on full data. Layer-20 peak holds but final-layer gap shrinks to +0.016 (layer selection is a minor win). |
| 14 | Per-task probe (`probe_per_task`) | Is the ceiling onset-specific? | instant_event_alert **0.789** (best), counting 0.78–0.82, narration 0.47 / step 0.62 (worst) | **Refutes onset-blindness.** Purest onset task separates *best*. Well-specified events all ~0.79; only open-ended commentary (narration/step) collapses. → uniform ceiling, points to #4 (untrained-for-firing). |

---

## 4. Ruling out query-conditioning (three independent tests)

| # | Experiment | What it tested | Result | Takeaway |
|---|---|---|---|---|
| 15 | Query-relevance drift (`diag_query_drift`) | Does conditioning decay across the stream? | MMDuet2 54.5% neg slope (p=0.23, n.s.); MMDuet 60.6% (p≈0 but first→last drop only 0.054) | **NOT decay.** Weak query-gating is static/uniform, not declining. |
| 16 | Re-prompt → probe separability | Does refreshing the query raise separability? | AUROC unchanged (layer 20: 0.790→0.784); no lift anywhere | Query **dilution/freshness is not the bottleneck.** Frozen re-injection doesn't help. |
| 17 | Trained query-cond probe (`probe_query_cond`) | Does trained query-conditioning help? | h_t 0.785; [h_t;q] concat 0.785; FiLM 0.738 (worse) | **Query info is not the missing signal.** Even trained concat/FiLM add nothing. Rules out the whole query-steering family. |
| 18 | Re-prompt firing metrics | Does re-prompt improve deployed firing? | P 0.093→0.146, F1 0.168→0.241, FP 14.5k→7.2k, R 0.863→0.707 | Apparent gain — but recall fell in tandem (fires less). Needed the matched-recall test to interpret. |
| 19 | Matched-recall PR (`matched_recall_test`) | Is re-prompt's precision gain real or just firing less? | Baseline PR curve **beats** reprompt at every recall (AUC-PR 0.225 vs 0.185); baseline@R=0.71 P=0.175 > reprompt 0.154 | Re-prompt gain is a (bad) operating-point shift — **thresholding the baseline beats it.** Re-prompting fires less, doesn't know better. Strictly dominated; dropped. |

---

## 5. Failure-symptom characterization (FP content)

| # | Experiment | What it tested | Result | Takeaway |
|---|---|---|---|---|
| 20 | FP content axes (`diag_fp_content`) | What are the false alarms? | Self-repeat 71–82% (@0.85) / 24% (@1.0 exact); unrelated-content 65–76%*; near-hits 23–33% | Over-firing decomposes into self-repetition (decision-fixable) + unrelated firing (representation-fixable) + near-hits (tolerance artifact). |
| 21 | Encoder check (MiniLM vs mpnet) | Is content-unrelatedness reliable? | TP-vs-GT cosine only 0.45 even with stronger encoder | *Content "unrelated" share UNRELIABLE — encoder can't judge this terse text. Needs LLM judge (pending). |
| 22 | FP query-relevance vs TP | Is firing query-gated at all? | MMDuet 56% of FP below median TP (≈chance); MMDuet2 67% | MMDuet near-indiscriminate; MMDuet2's RL gave partial query-gating. Robust across encoders. |
| 23 | Manual FP read | Hallucination vs real-but-irrelevant | FPs are coherent scene descriptions, not hallucinations. MMDuet forces query-topic on wrong frames; MMDuet2 drifts off-query into captioning | Two model-specific mechanisms, same precision failure: query-locked hallucination (MMDuet) vs query-abandonment (MMDuet2). Perception is fine; gating fails. |

---

## 6. Pending / in progress

| # | Experiment | What it tests | Status |
|---|---|---|---|
| 24 | MMDuet2 hidden-state probe | Does firing-RL training raise the separability ceiling? (tests #4) | Collection code ready (`inference_hs.py`); backbone-confounded so read shape/ceiling, not absolute |
| 25 | Visual-salience separability probe | Can h_t read query-independent visual salience? Localizes ceiling: query-gating gap vs general visual-info poverty | Proposed; needs frame-change salience labels joined to npz |
| 26 | LLM-judge FP content | Reliable related/unrelated + hallucinate-vs-abandon split | Pending (cosine unreliable) |
| 27 | Onset-vs-presence probe | (partly superseded by #14, which refuted clean onset-blindness) | Lower priority |

---

## Bottom-line diagnosis (current)

Over-firing is a **representational** failure: the firing-decision hidden state separates
queried-event frames from non-event frames at only **~0.79 AUROC** (layer 20, full data),
and this ceiling is **not** fixable by:
- decision rule / threshold (#7), nonlinear readout (#9), temporal windowing (#8),
- query re-injection (#16), trained query-conditioning / FiLM (#17), or re-prompting (#19),
- layer selection alone (#13, only +0.016).

It is **not** onset-specific (#14 — the purest onset task separates best) and **not** a
query-conditioning problem (#15–17, three independent nulls). The ceiling is roughly
**uniform across well-specified event types**, pointing to hypothesis **#4: the
representation was never trained to make firing-events separable** (shaped only by the
LM/next-token objective via stop_grad).

**Open crux:** whether finetuning (SFT/RL on the firing objective) can raise the ceiling,
or whether ~0.79 is a hard limit of the visual features. Experiments #24 (does MMDuet2's
RL-trained representation separate better) and #25 (can h_t even read visual salience)
are designed to resolve this and to choose between the two remaining fix directions:
**event-label representation training** vs **explicit change/onset input features**.
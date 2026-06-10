# Implementation Plan — Query-Conditioned Firing

*Companion to the design report (Rev. 2). Architecture-first sequencing:
counterfactual SFT restructuring, DTD, RL fine-tuning, and multi-query lanes
are explicitly deferred phases.*

---

## 0. Decisions already locked

| Decision | Choice | Rationale (see report) |
|---|---|---|
| Backbone | Qwen2.5-VL-7B-Instruct, stock | White-box access to hidden states + m-RoPE; strongest trainable open video backbone (§3.6) |
| Build strategy | Pretrained backbone + our modules; adopt recipes, not architectures | Ablation cleanliness; baseline parity on same backbone (§3.6) |
| Components (the entire diff) | Query sink (+ positional re-anchoring), carried scratchpad (with multi-depth taps), trigger head | §3.2–3.3 |
| Deferred | DTD, counterfactual restructuring, timing-aware/RL loss, F1 residual conditioning, fire-time scratchpad exposure, multi-query lanes | Confound-avoidance and scope control |
| Training mode (first) | Frozen backbone; train sink + scratchpad + head only | Cheap, isolates components |
| Base training data | TimeChat-Online-139K + VideoLLM-online-style silence supervision format, used as-is | Architecture phase needs *some* signal; restructuring comes later |

---

## Phase 0 — Harness and baselines (the floor everything is measured against)

**Goal:** a streaming train/eval harness on the Qwen backbone, with two
baselines reproduced on it.

1. **Streaming harness.** Port the VideoLLM-online interleave recipe onto
   Qwen2.5-VL: per-chunk visual ingestion (1 fps, fixed chunk size), query as
   text prefix, per-chunk `Silence`/`Response` supervision, KV-cache reuse
   across chunks, refractory/debounce at inference. Deliverable: a training
   loop and a streaming inference loop that share the chunking code path.
2. **Baseline A — token-driven (VideoLLM-online-style) on Qwen.** LM-head
   firing via special tokens, trained on the base SFT data. This is the
   canonical floor and the donor of all shared hyperparameters.
3. **Baseline B — salience-triggered (TimeChat-Online-style).** Scene-transition
   firing from inter-frame change (their drop-ratio signal can be computed
   without adopting DTD into our model: run the redundancy similarity check as
   a side statistic). Query-independent by construction — the swapped-query
   foil.
4. **Eval plumbing.** OVO-Bench forward-responding slice + StreamingBench
   proactive cells wired in; the two headline custom metrics implemented:
   firing-accuracy-vs-trigger-time curve, and swapped-query fire-time shift.

**Exit criteria:** Baseline A reproduces published-ballpark proactive numbers;
both headline metrics produce stable curves on baselines.

## Phase 0.5 — Diagnostics (cheap, runs in parallel with Phase 0 tail)

Two probes on Baseline A, before/while building components — they convert the
report's central premise from plausible to demonstrated and produce the
motivating figure:

- **Attention-mass probe:** attention mass on query-prefix tokens as a function
  of stream time, correlated with firing errors. Directly tests dilution vs.
  the length-OOD cliff (§2.1).
- **Re-injection probe:** periodically re-inject the query text; if timing
  accuracy recovers, dilution confirmed. Also serves as a weak baseline row.

**Exit criteria:** a figure attributing long-horizon firing failure to a named
mechanism (or honestly showing it's the length cliff — which still motivates
the sink's re-anchoring).

## Phase 1 — Components, in build order

Each component lands with its own unit-level validation before the next
starts. All training is frozen-backbone.

### 1.1 Query sink (smallest diff first)

- **Spec:** query tokens stored as a fixed, eviction-exempt KV block; visual
  tokens attend to it at every layer (it is ordinary prefix KV, just protected
  from any eviction/windowing). **Re-anchoring:** recompute the sink keys'
  RoPE rotation each chunk to a fixed small relative offset from the current
  position (constant work: sink length × layers).
- **Validation:** attention-mass probe rerun — mass on the sink should be flat
  in stream time where the plain prefix decayed. Long-horizon firing curve vs.
  Baseline A.
- **Risk:** m-RoPE's 3D positions make re-rotation slightly fiddlier than text
  RoPE; isolate in a unit test against a reference implementation.

### 1.2 Trigger head (before the scratchpad — it's the consumer)

- **Spec:** `p_fire = σ(g(h_t, q_fixed))` initially (scratchpad input added in
  1.3). `g` = small MLP/transformer block; `q_fixed` = pooled stored query
  embedding (computed once at query registration, never re-read through the
  prefix). Trained with per-chunk BCE + class reweighting for now
  (timing-aware loss is a deferred upgrade). LM head no longer decides firing;
  generation triggered by `g` crossing threshold, with refractory.
- **Validation:** matches or beats Baseline A overall; the q_fixed-ablated
  head (g(h_t) only) must be measurably worse on swapped-query shift —
  if it isn't, the base data isn't exercising query-dependence and Phase 3
  moves up in priority.

### 1.3 Carried scratchpad (the main build)

- **Spec (per report §3.2):** 8–32 learned tokens; initialized from the query
  embedding at registration; per chunk, cross-attends (read-only) to hidden
  states tapped at every 4th layer; gated convex update
  `s_t = (1−γ_t)s_{t−1} + γ_t Δ(s_{t−1}, H_t, q_fixed)`, zero-init gate,
  layernorm on s_t; reset-on-fire (except running tallies). Head becomes
  `g(h_t, q_fixed, s_t)`.
- **Training mechanics:** truncated BPTT over a small window of chunks
  (start: 4–8 chunks); state-carryover curriculum — progressively longer
  carried sequences across training; detach state at window boundaries.
- **Validation (these ablations ARE the paper's evidence):**
  1. carried vs. blank-each-chunk scratchpad — on accumulation/duration
     queries, at increasing horizon (crossover at long horizon = drift
     signature; gap growing with horizon = persistence thesis confirmed);
  2. multi-depth taps vs. final-layer-only readout;
  3. gate-open statistics over distractor stretches (should be sparse).
- **Fallback knob:** if frozen-backbone underperforms, unfreeze the top
  k backbone layers (k ∈ {2, 4, 8}) before touching anything else.

### 1.4 Integration pass

- Full stack: sink + scratchpad + head, one config; component-removal matrix
  (each of the three off, pairs off) on a fixed eval split.
- Verify the clean-ablation property mechanically: with all components
  removed, logits are byte-identical to stock Qwen2.5-VL on the same input.

**Phase 1 exit criteria:** the full stack beats Baseline A on (a) flatness of
the firing-accuracy-vs-time curve and (b) swapped-query fire-time shift, and
beats Baseline B on (b) by a wide margin; component matrix shows each part
contributes.

## Phase 2 — Scale-up and characterization

- Longer training streams (curriculum to ≥30-min effective horizon via
  carryover), full eval suite (OVO-Bench, StreamingBench proactive cells,
  ESTP-F1 if harness time permits, OmniPro).
- Per-query-type breakdown: instantaneous / relational / accumulation /
  duration / precondition — the scratchpad should matter only for the latter
  three (a prediction worth reporting either way).
- Decision gate: is the swapped-query shift saturating without counterfactual
  data? If yes, Phase 3 is confirmation; if no, Phase 3 is rescue.

## Phase 3 — Supervision upgrades (deferred; activate after Phase 2 gate)

1. Counterfactual restructuring of existing annotations (swapped-query pairs,
   q_null distractor streams, enter/exit temporal-verb hard negatives) — see
   report §3.4. Scripted over Ego4D moments / dense captions /
   TimeChat-Online-139K; no new annotation.
2. Timing-aware loss replacing reweighted BCE; GRPO on a timing reward as a
   stretch comparison.
3. Re-run the Phase 1.3 ablation matrix on the upgraded data — the
   architecture-vs-supervision decomposition (how much of the failure each
   fixes) is a headline result in itself.

## Phase 4 — Multi-query (deferred; design already in report §4)

- Small-N shared sequence with per-query sinks + firing tokens
  `[Silence, Q1…QN]`; per-query scratchpads.
- N-sweep (report §6) → lane/no-lane crossover; lanes (top-K routing, prefix
  caching via vLLM/SGLang) only if the crossover demands them.

## Set aside (revisit only after core results)

- **DTD** — efficiency module; would confound long-horizon ablations now.
  Post-core, test as an add-on (expected: slows dilution; weak for duration
  queries, which the scratchpad covers).
- **F1 residual query→visual conditioning** — representation fix, incremental;
  add as a plus-one ablation row late.
- **Fire-time scratchpad exposure to generation** — content-quality bonus,
  orthogonal to timing; one ablation row in Phase 2/3.
- **Per-layer query prefix injection** — stronger structural conditioning if
  the sink alone proves insufficient at depth; keep on the shelf.

## Risk register (live)

| Risk | Detector | Mitigation |
|---|---|---|
| Re-anchoring breaks m-RoPE assumptions | Unit test vs. reference; perplexity sanity on stock tasks | Fall back to position-free sink keys |
| Scratchpad long-horizon drift | Carried-vs-blank crossover at long horizon | Carryover curriculum; distractor stretches; head learns to discount (floor ≈ no-scratchpad) |
| Base SFT data doesn't exercise query-dependence | q_fixed-ablation shows no gap (Phase 1.2) | Pull Phase 3 restructuring forward |
| Trigger over/under-firing collapse | Precision/recall at fixed threshold sweep | Joint tuning of threshold + negative-sampling ratio; benchmarks penalize over-triggering |
| Truncated BPTT instability | Loss spikes at window boundaries | Shorter windows, state detach + layernorm, lower scratchpad LR |
| Frozen backbone insufficient | Phase 1.3 underperformance | Unfreeze top-k layers (knob, not redesign) |

## Evidence map (what each experiment proves)

| Claim | Experiment |
|---|---|
| Long-horizon failure is query-binding decay | Phase 0.5 probes |
| Non-decaying query path fixes timing | Sink ablation; curve flatness vs. Baseline A |
| Firing is query-conditioned, not salience | Swapped-query shift vs. Baseline B (and StreamingClaw-style fixed tokens, by construction) |
| Persistent state is necessary for stateful queries | Carried-vs-blank × query-type × horizon |
| Multi-depth readout matters | Taps vs. final-layer ablation |
| Architecture permits, supervision compels | Phase 3 decomposition |

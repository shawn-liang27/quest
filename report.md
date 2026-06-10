# Query-Conditioned Firing for Proactive Streaming VideoLLMs

*Design report — single-query and multi-query proactive monitoring*

**Rev. 2 (June 2026).** Changes from Rev. 1: (a) the positional-decay argument in
§2.1 is replaced with the sounder dilution + out-of-distribution-length framing,
and a causal-gap diagnostic is added; (b) the predicate-state register (§3.2) is
replaced by a **carried scratchpad** with multi-depth taps, removing the
state-annotation dependency; (c) the trigger head (§3.3) now reads the
scratchpad; (d) supervision (§3.4) is re-scoped to data *restructuring* and
deferred to a later phase; (e) substrate decision added (§3.6): Qwen2.5-VL-7B,
recipes adopted from VideoLLM-online / TimeChat-Online, DTD set aside; (f)
StreamingClaw and TimeChat-Online added to baselines (§2.2). A separate
implementation plan document accompanies this report.

---

## 1. Problem Statement

A proactive streaming VideoLLM receives a standing query *q* at time 0 and then
observes a video stream frame-by-frame. At each timestep *t* it must make two
decisions:

1. **When to respond** — the *trigger* decision (typically emitting a
   `Silence`/`EOS` token to stay quiet or a `Response` token to speak).
2. **What to say** — the response generation, conditioned on a fire decision.

Formally, the trigger at frame *t* is a binary predicate approximating: *given
the standing query q and everything observed up to t, is the condition specified
by q now satisfied?* The task is proactive monitoring / alerting — *q* names a
condition ("alert when the doorbell rings," "tell me when the pot boils over,"
"notify me when a person in red enters") — and correctness requires both
**temporal accuracy** (fire within a small window of the true event) and
**content correctness**.

**The core property that must hold:** *the firing time must be a function of the
query.* The same video stream paired with a different query should fire at
different moments. This is the defining requirement of proactive monitoring and
the thing current systems fail to guarantee.

**Scope extension.** Beyond a single standing query, a practical monitor must
handle a **set** of standing queries *Q = {q₁, …, q_N}* and alert on whichever
condition fires, responding only about the event it actually observed (not
emitting a structured status report every timestep). This introduces a
multiplexing problem treated separately in §4.

---

## 2. Current Limitations and Baselines

### 2.1 Why current systems fail

**The firing decision is not query-conditioned in any stable way.** In the
prevailing design the query enters as a text prefix at position ~0 of the KV
stream, and the `Silence`/`Response` token is produced by the same
autoregressive head that generates language, attending over
`[query ⊕ all visual tokens]`. The query's influence on the trigger therefore
routes entirely through attention over a growing sequence.

**Attention dilution causes query-grounding to decay with stream length.** As
visual tokens accumulate, the following effects compound:

- **Softmax dilution (mathematically unavoidable):** visual tokens vastly
  outnumber the fixed query tokens, and the softmax denominator grows
  monotonically with stream length, so attention mass on the query shrinks.
- **Out-of-distribution length extrapolation:** once the stream exceeds the
  relative distances seen in training, RoPE attention degrades sharply (the
  motivation for NTK/YaRN scaling). The sharp collapse beyond ~180s in the
  benchmark evidence below is more consistent with this distribution-shift
  cliff than with smooth positional decay.
- The firing decision drifts from query-conditioned salience ("did *the queried
  thing* happen?") toward generic salience ("did *something interesting*
  happen?").

*Dropped from Rev. 1:* "positional decay down-weights the now-distant query."
This conflicts with the attention-sink literature — initial tokens in
autoregressive LMs retain disproportionate attention mass (why StreamingLLM
pins them), and lost-in-the-middle is U-shaped, favoring the beginning. A
position-~0 query is comparatively *privileged*; dilution and OOD length are
the defensible mechanisms.

**Causal gap and required diagnostic.** The long-horizon benchmark numbers are
*consistent with* query-binding decay but do not isolate it — KV
eviction/compression, visual memory degradation, and train/test length mismatch
are confounders. Before (or in parallel with) building, run two cheap probes on
an existing baseline: (i) **attention-mass probe** — measure attention mass on
query tokens as a function of stream time and correlate with firing errors;
(ii) **re-injection probe** — periodically re-inject the query text; if timing
accuracy recovers, dilution is confirmed. These convert the central premise
from plausible to demonstrated and yield the motivating figure for the paper.

**This is empirically the dominant failure mode at long horizons.** Recent
benchmarks show proactive performance retaining only ~37% of short-range levels
at long range, with sharp collapse beyond ~180s — precisely the regime where the
query-to-trigger binding has decayed most, and precisely the alert/monitor task
family that depends on sustaining a specific condition over time.

**Perception-side fixes do not reach the firing decision.** Cross-attending
visual tokens to the query *before* the backbone (weighting query-relevant
content at write time) fixes *what the backbone sees* and preserves reasoning
inside the LLM — but it is a representation fix. The trigger logit is still
produced by the generic head over the diluting sequence, so better-weighted
inputs still feed a query-agnostic firing criterion. *Right tokens lit up, wrong
decision path.*

**Cross-attention conditioning that bypasses the backbone sacrifices
reasoning.** Making the trigger a function of query·visual similarity *outside*
the LLM (the propose-match / cosine-trigger approach) gives a query-conditioned
signal but a memoryless, non-reasoning one. Cosine similarity answers "does this
frame resemble the proposal," not "given everything seen, is the condition
satisfied" — it cannot handle relational conditions (state transitions),
maintained state (counts), or abstract conditions that don't reduce to frame
resemblance.

**The supervision never teaches query-dependence of timing.** Standard trigger
training pairs a video with one query and the annotated fire timestamp. The head
is never shown that the same video under a *different* query should fire
*elsewhere*. So even an architecture capable of query-conditioned timing is not
forced to learn it.

**The gap, stated tightly:** what-enters-the-backbone and what-the-backbone-says
can be made query-aware, but *when-it-fires* is decided on a path where the
query's influence decays — and the training signal never requires that influence
in the first place.

### 2.2 How the limitation maps onto existing approaches (baselines)

| Approach | Trigger mechanism | What it isolates as a baseline |
|---|---|---|
| **VideoLLM-online** | `Silence`/`Response` special tokens via the LM head | "Trigger via diluting autoregressive head" — canonical floor |
| **MMDuet / MMDuet2** | Token-driven response/silence head | The token-driven family the trigger-head approach directly improves on |
| **Dispider** | Disentangled perception / decision / reaction | "Decision separated from generation" — closest architectural cousin |
| **StreamForest** | Persistent event memory, query-agnostic firing | "Good memory, but firing not query-conditioned" |
| **Em-Garde** | Propose-match: query→visual cosine on a static MLLM | "Query-conditioned but no backbone reasoning" — key conceptual baseline |
| **LiveStar** | Token-driven | Near-zero on counting/monitoring in online mode — floor, not competitive |
| **TimeChat-Online** | Fires at scene transitions detected from DTD drop-ratio dips | "Generic visual salience firing, query-independent by construction" — cleanest salience foil; shares our backbone, so perfectly controlled |
| **StreamingClaw (StreamingProactivity)** | Scenario-specific trigger tokens in a decoupled sub-agent | "Closed-set event-token detection, decoupled from backbone reasoning" — contrast: our firing tokens bind dynamically to arbitrary natural-language queries at inference time, and the head reads full backbone state |

**Positioning vs. StreamingClaw (required in related work).** Their trigger
tokens are fixed at training time (token *i* permanently means event-type *i*)
and are explicitly designed to *decouple* perception from downstream reasoning.
Ours differ on three axes: (1) the fire decision reads the full backbone hidden
state (in-backbone reasoning at fire time); (2) firing tokens bind to arbitrary
runtime queries (open-vocabulary), not a pretrained event catalog; (3) we make
non-decaying query influence a structural property (sink + `q_fixed` +
scratchpad) rather than a system-level one (reminder nodes, re-prompting). The
swapped-query metric (§5.4) is the sharpest empirical separator: fixed
scenario tokens cannot track a swapped query by construction.

### 2.3 A note on attention fragility (evidence-checked)

The "queries starve each other" framing is **overstated** as a binary failure.
The long-context literature shows multi-target tracking degrades *gracefully*
with the number of simultaneous targets, not off a cliff:

- Frontier models report near-perfect **single**-needle recall across up to 1M
  tokens (≈99–100%), so the attention budget is not so thin that a handful of
  queries collapses it.
- But there is a **real, measured penalty that scales with N** and compounds
  with reasoning: multi-needle retrieval degrades as more facts are requested,
  degrades further when the model must *reason* over them, and drops *more
  steeply* than single-needle as context grows.
- **Placement matters** — mid-context relevant spans lose 30–50% accuracy
  ("lost in the middle"), which is exactly what a pinned, non-decaying query
  position fixes.

**Implication:** for a handful of standing queries with correct visual
highlighting and pinned placement, pretrained attention's native multi-tracking
should carry a *single shared sequence*. Dilution is a smooth cost that scales
with N, context length, and reasoning depth — and since this task is
long-horizon and reasoning-heavy, the effective N-ceiling is *lower* than
generic retrieval benchmarks suggest. **This crossover should be measured, not
assumed** (see §6). Caveat: all of this evidence is on *text* needles; visual
tokens are denser and less separable, so visual multi-query dilution may behave
somewhat worse — another reason to run the sweep on video directly.

---

## 3. Proposed Solution — Single Query

Design principle: **inject the query into the firing decision through paths that
do not decay with sequence length, while keeping reasoning inside the backbone
and leaving response generation untouched.** Three fronts.

### 3.1 Front 1 — Condition visual features before the backbone

Modulate incoming visual tokens by relevance to *q* via cross-attention / gating
*at write time*, so the backbone reasons over query-biased representations.

- **Residual, not replacing:** `v'_t = v_t + α · CrossAttn(v_t, q)`. The
  unconditioned `v_t` must survive into the backbone, because relational queries
  ("has the person *left*") need the backbone to reason over unconditioned
  visual state, not just the query-aligned slice. The gate must *emphasize*,
  never *filter* — a filtering gate here silently degrades the firing decision
  in Front 3.

This fixes *what enters the backbone*; it does **not** by itself make *when it
fires* query-conditioned.

### 3.2 Front 2 — Persistent query conditioning (anti-decay)

- **Query sink:** store the query as a **fixed, eviction-exempt,
  decay-exempt** KV block — an attention sink that visual tokens always attend to
  at full weight regardless of stream length. This is the structural guarantee
  that query→visual conditioning is *constant*, not slowly degrading. (Periodic
  text re-injection is the weak version; the pinned sink is the principled one
  and costs no extra tokens over time. It also directly counters the
  "lost-in-the-middle" placement penalty.)
  - **Positional re-anchoring (added):** eviction-exempt is not
    length-OOD-exempt. Re-rotate the sink's keys to a fixed small relative
    offset from the current frame (or give them position-free encoding), so
    the query stays at an *in-distribution relative distance* forever as the
    stream grows. Nearly free; closes the gap between the dilution argument
    and the length-extrapolation argument in §2.1.

- **Carried scratchpad (replaces Rev. 1's predicate-state register):** a small
  set of learned tokens (8–32) that live **outside the backbone's softmax** and
  persist across chunks. Per chunk:

  1. **Query-initialized:** the scratchpad is initialized once from the query
     embedding, making it a query-conditioned register ("progress toward *this*
     condition"), not a generic compressed history.
  2. **Multi-depth taps:** the scratchpad tokens cross-attend (as queries) to
     backbone hidden states tapped at a few depths (e.g., every 4th layer).
     Taps are **read-only** — visual tokens never attend to the scratchpad, so
     the backbone's computation and the response path are byte-identical to the
     base model when trigger components are removed (the clean-ablation
     property). Rationale: the final-layer state is optimized for next-token
     prediction and discards information mid layers retain; a fire decision
     needs both low-level perceptual change and high-level semantics.
  3. **Gated convex update:** `s_t = (1−γ_t)·s_{t−1} + γ_t·Δ(s_{t−1}, H_t,
     q_fixed)`, with γ learned and zero-init-biased closed, plus layernorm on
     `s_t`. Properties: updates are sparse in practice (the gate opens on
     query-relevant events, not every chunk); old content is exponentially
     forgotten, not accumulated (no random-walk noise growth); magnitude is
     bounded. Every update re-reads `q_fixed`, continually re-grounding the
     state in the condition (anti-drift).
  4. **Reset-on-fire:** once fired, the condition is consumed — reset the
     episode state (but not running tallies for continuing "every time X"
     queries). Pairs with the refractory/debounce in §3.5.

  - **Needed** for duration / accumulation / precondition queries (boiled *for*
    2 min; the *3rd* occurrence; "after X, alert on Y") — this state is exactly
    what KV eviction/pruning/window-sliding will not reliably retain over
    minutes, and DTD-style redundancy dropping actively removes (static content
    is dropped; duration queries are *about* static content).
  - **Near-memoryless for instantaneous alerts** ("doorbell rings"): the
    learned gate stays closed and `h_t` carries the decision — no hand-set
    policy needed.
  - **Why this replaces the register:** the Rev. 1 register required
    state-annotated supervision (flagged as the top feasibility risk). The
    scratchpad trains **end-to-end from fire timestamps alone** — gradients
    flow from the timing loss through the trigger head into the scratchpad and
    its cross-attention; the slots learn what to store. Counterfactual
    restructured data (§3.4), when added, forces the slots to track the
    *queried* condition specifically.
  - **Known failure modes and mitigations:** (i) *semantic drift* — mitigated
    by re-reading `q_fixed` at every update and by reset-on-fire capping
    episode length; (ii) *train/test horizon mismatch* — the gate has never
    been tested against an hour of temptation if trained on minutes; mitigate
    with a state-carryover curriculum (progressively longer carried sequences
    under truncated BPTT) and long irrelevant distractor stretches in training
    streams (an explicit stay-closed signal, which null-query counterfactual
    pairs provide nearly for free). Floor guarantee: the head reads the
    scratchpad as one feature among three, so a degraded scratchpad bounds
    performance near the no-scratchpad baseline rather than below it, provided
    training included streams long enough for the head to learn when to
    discount it. The carried-vs-blank ablation at increasing horizon is the
    direct detector of noise accumulation (crossover = failure signature).

### 3.3 Front 3 — The firing signal

Replace reliance on the generic LM head for the trigger with a **dedicated
learned trigger head** whose input re-supplies the query every frame from a
stable slot:

```
p_fire(t) = σ( g( h_t , q_fixed , s_t ) )
```

- `h_t` carries reasoning and accumulated, query-biased context (Fronts 1–2) —
  fresh and full-reasoning, but final-layer-only and subject to the dilution
  the design targets.
- `q_fixed` is pulled from a stored query embedding, not read through the
  decaying prefix — so its contribution to the firing logit *cannot decay* with
  stream length.
- `s_t` is the carried scratchpad (Front 2) — multi-depth and persistent but
  low-capacity. `h_t` and `s_t` are complementary, not redundant; the head
  learns to weigh them per query type (same feature-not-multiplier philosophy
  as the relevancy decision below).
- In the multi-query case (§4), this head produces the logits over
  `[Silence, Q1, …, QN]`.

**The scratchpad feeds the trigger head, not the LM head.** Routing `s_t` into
the shared LM vocabulary head would lose the dedicated-path story and let
trigger training perturb generation. The one sanctioned exception: **at fire
time only**, the scratchpad states may be exposed to generation as context —
for state queries it carries exactly what the response needs ("the *3rd*
occurrence"), which the diluted KV may no longer surface. Optional, ablatable,
zero cost while silent.

**Relevancy-as-a-feature, not a multiplier.** If a query·visual similarity
signal is used, it is *concatenated into* `g`'s input and the head *learns* how
much to trust it per query type — **not** combined as `P_reasoning × S_relevance`.
The multiplicative gate is rejected: at the true firing moment for a relational
condition, single-frame similarity is *low*, so a product would suppress the
*correct* fire; the two scores are not calibrated to be multiplied; and it
re-imports the very memoryless limitation the design avoids.

**Response generation is unchanged** — once a fire is decided, the existing LM
head generates over the (query-biased) context.

### 3.4 Supervision (deferred phase — restructuring, not curation)

Architecture *permits* query-conditioned timing; **counterfactual SFT compels
it.** Status: **deferred until the architecture phase is complete** (see the
implementation plan). Two scoping corrections from Rev. 1:

- This is data **restructuring, not curation**: multi-event annotated corpora
  (Ego4D moment queries, dense-caption datasets, TimeChat-Online-139K) already
  provide Event A at t₁ / Event B at t₂ per video; pairing them with swapped
  queries is scriptable. No new video collection or annotation.
- The **state-annotation requirement is dropped**: the scratchpad (§3.2) trains
  end-to-end from fire timestamps, so per-frame state labels are no longer
  needed. Rev. 1's hardest data dependency is gone.

The restructuring scheme, when this phase activates — same video, swapped
queries, shifted ground-truth fire times:

- For a video with Event A at t₁ and Event B at t₂:
  - `(V, q_A)` → `Silence` until t₁, then `Fire`.
  - `(V, q_B)` → `Silence` through t₁, `Fire` at t₂.
  - `(V, q_null)` (event never occurs) → `Silence` indefinitely — doubles as
    the scratchpad's stay-closed gate signal over long distractor stretches.
- **Hard negatives on temporal logic:** `q_enter` ("alert when the person
  enters") vs. `q_exit` ("alert when the person leaves") over heavily-overlapping
  visual tokens — forcing the loss onto the temporal verb, not visual
  resemblance.
- **Timing-aware loss**, not per-frame binary cross-entropy (which is dominated
  by the silent majority class): reward firing inside the ±window, penalize
  silence-through-event *and* premature/late fire. (RL-style fine-tuning — e.g.
  GRPO on a timing reward — is a candidate alternative/complement; deferred.)

Note: the architecture phase still requires *some* training signal for the new
components — base streaming SFT data used as-is (TimeChat-Online-139K +
VideoLLM-online-style silence supervision) suffices for that; the
counterfactual restructuring is the deferred part.

### 3.5 Single-query composition and constraints

`shared visual encode → residual query-conditioning (F1, optional first pass) →
backbone with pinned query sink (F2) → carried scratchpad with multi-depth taps
(F2) → learned trigger head g(h_t, q_fixed, s_t) (F3)`, trained first on base
streaming SFT, later with counterfactual + timing-aware restructuring (§3.4).

Cross-front constraints:

- Front 1's gate **must stay residual** — Front 3 needs unconditioned visual
  reasoning in `h_t`.
- Front 2's scratchpad is **read by** Front 3's head — design it as the
  head's input from the start, not bolted on.
- The scratchpad's taps are **read-only**: removing sink + scratchpad + head
  must recover the base model exactly (ablation cleanliness is the evidence
  strategy).
- Gradient path: timing loss → `g` → scratchpad + cross-attention parameters,
  optionally into the backbone via the taps. **Frozen backbone first** (train
  sink + scratchpad + head only); unfreeze upper layers only if it
  underperforms.

Add a **per-condition refractory/debounce** so a single event doesn't spam
alerts across adjacent frames (interacts with reset-on-fire, §3.2).

### 3.6 Substrate decision (added in Rev. 2)

**Build from a clean pretrained VLM plus our modules; adopt scaffolding, not
architectures.**

- **Backbone: Qwen2.5-VL-7B-Instruct.** Strongest open video-language backbone
  at trainable scale; native interleaved video-text support and m-RoPE; the
  sink, taps, and trigger head need white-box access to hidden states and the
  RoPE implementation, which it gives cleanly. Frozen-backbone first
  experiments fit modest compute. (Re-evaluate Qwen3-VL maturity at project
  start.)
- **Adopt the recipe, not the model, from VideoLLM-online:** interleaved
  streaming format, per-chunk silence supervision, refractory logic, eval
  protocol — reimplemented on the Qwen backbone as the token-driven baseline.
  Same backbone + same data ⇒ every gap is attributable to our components.
- **Do not build on Em-Garde:** its propose-match core is the
  decoupled, reasoning-light trigger family this report argues against;
  modifying it means ripping out its core mechanism and inheriting its
  constraints. It belongs in the baseline table.
- **TimeChat-Online:** take the **dataset** (139K streaming QA, 11-min videos,
  negative samples) as base SFT corpus; take its scene-transition trigger as
  the **salience-firing baseline** (it fires on DTD drop-ratio dips —
  query-independent by construction, the exact foil for the swapped-query
  metric). **DTD itself is set aside**: it is an efficiency module, orthogonal
  to the contribution; adding it now confounds long-horizon ablations
  ("scratchpad or DTD?"). Revisit post-core-results as an optional add-on
  (expected synergy: ~80% token reduction slows softmax-denominator growth).
- **Keep everything else stock.** No exotic visual encoders or memory modules.
  The diff against a known backbone is exactly four things: sink (+
  re-anchoring), scratchpad (+ taps), trigger head, and (optional, later)
  residual F1 conditioning. The narrower the diff, the stronger every result
  reads.
- **Serving (later, multi-query phase):** vLLM/SGLang with prefix caching.

---

## 4. Proposed Solution — Multi Query

Goal: handle a standing set *Q = {q₁, …, q_N}*, fire on whichever condition
triggers, and report only the event observed — *without* emitting a structured
status report every timestep.

### 4.1 The new problems multi-query introduces

1. **State-machine multiplexing** — one forward pass must track N independent
   temporal state machines in the same vector space; features can tangle
   (superposition interference) as N grows.
2. **Credit assignment** — the system must attribute a fire to the *correct*
   query; naïve relevancy gating can validate query *i*'s reasoning with query
   *j*'s relevancy (cross-validation mismatch).
3. **Intra-query dilution** — N queries in one softmax compete for attention
   mass (real but *graceful*; see §2.3).
4. **Combinatorial SFT** — the model must fire on q_i while q_j is half-complete,
   q_k just finished, and q_l is visually distracting.

### 4.2 Substrate: route-then-isolate

The governing choice is **one shared sequence vs. N per-query "lanes."** A
**lane** = separated *attention/reasoning state per query* over a **shared
visual computation** — *not* N independent inferences.

- The expensive part — the **visual stream — is encoded once**, one growing
  visual KV cache, never duplicated.
- Per-lane attention over that shared cache is a **batch dimension**: backbone
  run with batch size K, each element = `[q_i sink] + [shared visual KV]`. The
  visual KV is **prefix-shared** across the batch (supported natively by
  vLLM/SGLang), so cost is *one visual pass + K lightweight query-conditioned
  reads*, not K full passes.
- **"Lane i's query never shares a softmax denominator with lane j's"** means: in
  lane i's attention, only q_i's keys are present in the softmax sum — so q_i
  gets undiluted mass regardless of how many other queries exist. This makes
  **intra-query dilution structurally zero** for the isolated path, *without* the
  block-diagonal-mask hack (which still leaves visual→query dilution intact).

### 4.3 The recommended policy (evidence-driven)

Because §2.3 shows multi-tracking degrades *gracefully* for small N with good
placement:

- **Small N (default):** **single shared sequence** with all queries **pinned as
  sinks** (Front 2). No batching. Let native multi-tracking do the work. Credit
  assignment via **query-specific firing tokens** — the trigger head outputs over
  `[Silence, Q1, …, QN]`, so breaking silence *requires* committing logit mass to
  a specific query identifier.
- **Large N / past the crossover:** **top-K routing → K lanes.** A lightweight
  router scores the frame against the standing list, selects the K plausibly-
  relevant queries (K ≈ 3–8), and only those run as isolated lanes (Fronts 1–3
  per lane). N can be large; K and therefore cost/latency stay bounded.
  - **Hard boundary:** the router decides *who reasons*; the **learned trigger
    head decides whether to fire.** A memoryless router must **never** make the
    final fire decision (the "wake-word" trap caps recall at the dumb
    component).
  - **Hysteresis:** once a query is *near* threshold, keep it "warm" for a few
    seconds so the router does not drop a query immediately before its event
    begins.

Per-lane isolation gives **free credit assignment** (which lane crossed
threshold = which query fired; simultaneous fires → report both, which is
correct for "any of these").

### 4.4 Multi-query state and supervision

- **Per-lane / per-query carried scratchpad** — the maintained-progress
  memory of §3.2, one per query (each initialized from its own q_i). This
  matters *more* in multi-query (it stays stable where attention decays across
  the long horizon) and is the piece most often omitted.
- **Distractor-injected counterfactual SFT:** e.g. input
  `Video + [Q1] dog barks, [Q2] person leaves, [Q3] lights off`; target
  `Silence … <Q2> "The person has left the room."` — explicitly penalizing fires
  on Q1/Q3 and teaching per-query isolation of the firing decision.

### 4.5 Cost note

Batching is **not free**: KV-activation memory scales with K, and K lanes each
running full backbone depth over the shared cache is real compute — whether it
holds at target FPS depends on model size and K. **Fallback if too slow:** share
the lower backbone layers (largely query-agnostic visual feature-building) and
branch into per-lane attention only in the upper layers — cheaper, at some cost
to how early the query conditions the reasoning. This is the compute/quality
knob.

---

## 5. Benchmarks and Baseline Models

### 5.1 Benchmarks

| Benchmark | Why use it | Caveat |
|---|---|---|
| **OmniPro (Online / F1)** | Closest to the problem: autonomous firing, alert/monitor/count subtasks, **penalizes over-triggering** | Single standing query per sample |
| **ESTP-Bench (+ ESTP-F1)** | Best at isolating *timing quality* — just-in-time, penalizes under/over-extended answers | Egocentric focus |
| **OVO-Bench (Forward Active Responding: CRR/SSR/REC)** | Standard proactive-timing slice; comparability with prior work | Dense-probing protocol is biased ("always-yes ≈ 60%") — report, don't lead with it |
| **StreamingBench (proactive-output acc.)** | Broad, widely reported, good for situating | Same probing-bias caveat for timing cells |

**Multi-query gap.** No existing benchmark natively evaluates "alert when *any of
N* events fire" (OmniPro is single standing query; OmniMMI is proactive but
single-response). The multi-query evaluation must be **constructed** — compose
OmniPro-style videos with multiple standing conditions and per-query firing
annotations. This construction is itself part of the contribution and is where
the N-sweep (§6) lives.

### 5.2 Baseline models

- **Open-source streaming (core set):** StreamForest (open SOTA on proactive
  cells), Dispider, MMDuet/MMDuet2, VideoLLM-online; LiveStar as a floor.
- **Targeted method baseline:** **Em-Garde** — the query-conditioned-but-no-
  reasoning point to differentiate from.
- **Proprietary reference ceiling:** Gemini (Flash tier), GPT-4o — reference
  points, not architectural baselines (OmniPro: Gemini ≈40% probe vs ≈22% best
  open-source).

### 5.3 Ablation mapping

| Baseline | Isolates | Method should win on… |
|---|---|---|
| VideoLLM-online / MMDuet | Trigger via diluting AR head | Late-trigger cells (flatness over time) |
| Em-Garde | Query-conditioned, no reasoning | Relational / state conditions (state-monitor, conditional-alert); *match* on simple alerts |
| StreamForest | Persistent memory, query-agnostic firing | Flat firing-accuracy-vs-trigger-time curve |

### 5.4 Headline metrics (foreground these, not aggregate F1)

1. **Firing-accuracy-vs-trigger-time curve (flatness)** — the decisive metric;
   measures whether query-grounding holds at long horizon. A method that raised
   average F1 without flattening this curve would *not* have solved the problem.
2. **Query-dependence of firing time** — under swapped-query evaluation on
   identical video, fire times should *shift to track the new condition*;
   baselines cluster around generic-salience moments.
3. **Per-query firing precision/recall as a function of N** (multi-query) —
   specifically on the queries that are *not* firing, to detect drop / spurious
   fire. This curve reveals the lane/no-lane crossover.

*Caveat: leaderboards move fast. "StreamForest = open SOTA" and specific OmniPro
numbers were accurate as of June 2026; re-verify when writing related work.*

---

## 6. Validation Plan — the N-sweep

The lane/no-lane decision and the dilution-severity question are **empirical**.
Run a controlled sweep:

- **Hold fixed** at the task's *real* values: context length (long horizon),
  reasoning difficulty (relational/state conditions), visual highlighting
  quality.
- **Vary** N (number of simultaneous standing queries).
- **Measure** per-query firing precision/recall — **especially for the
  non-firing queries** (do they get dropped or spuriously fire when attention
  concentrates on the active one?).
- **Output:** the N at which firing accuracy on non-active queries begins to
  drop = the crossover where lanes start to earn their batching cost. Expect it
  *lower* than text-retrieval benchmarks imply (long-horizon + reasoning-heavy +
  denser visual tokens).

**Prototype-first risks** (most likely to not work, test early):

1. **Prefix-sharing a *growing* cache across a batch** — clean in PyTorch,
   fiddly in production inference engines; requires query-sink-first layout.
   Validate against the serving stack before committing to lanes.
2. **Top-K router recall ceiling** — a dropped query right before its event
   begins is unrecoverable; validate hysteresis/warm-keeping.
3. **Scratchpad long-horizon drift** — the gated update is noise-resistant by
   construction, but the train/test horizon mismatch is untested until the
   carryover curriculum runs; the carried-vs-blank ablation at increasing
   horizon is the detector (crossover = noise-accumulation signature).
4. **Trigger-head over/under-firing collapse** — tune fire threshold and
   negative-sampling ratio *jointly*; over-triggering is explicitly penalized by
   the benchmarks.

---

## 7. Summary

| Layer | Single query | Multi query |
|---|---|---|
| **Visual conditioning (F1)** | Residual query→visual cross-attn at write time (optional, later) | Same, applied per active lane after routing |
| **Persistence (F2)** | Pinned, re-anchored query sink + carried scratchpad (multi-depth taps, gated update, reset-on-fire) | Per-query sinks (shared seq) or per-lane (routed) + per-query scratchpad |
| **Firing (F3)** | Learned head `g(h_t, q_fixed, s_t)`; relevancy as feature | Query-specific firing tokens (small N) / per-lane heads (large N); router gates *who reasons*, never *whether to fire* |
| **Supervision** | Base streaming SFT now; counterfactual swapped-query + timing-aware restructuring deferred | + distractor-injected counterfactuals (deferred) |
| **Substrate** | Qwen2.5-VL-7B, frozen-first; single sequence | Shared visual encode → top-K route → K isolated lanes (vLLM/SGLang prefix caching) |

The throughline: **what enters the backbone and what it says are already
query-aware; the contribution is making *when it fires* query-conditioned through
non-decaying paths (pinned re-anchored sink + injected `q_fixed` + carried
query-initialized scratchpad) and — in a later phase — a training signal that
compels query-dependent timing; extended to a set of standing queries by
routing-then-isolation, with the dilution crossover measured rather than
assumed.**

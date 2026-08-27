# I-CE (I2CE) — Complete Model Specification and Experimental Record

**Scope.** This document specifies the I-CE recipe as implemented in
`Pod/w9_cv_worker.py`, gives the experimental evidence behind each design
decision, states the capability it trades away (tag F1), and records every model
variant tried in the w9 campaign together with its failure mode.

**Reading the numbers.** Headline results are **five-fold cross-validation** at
the 4,096-sentence anchor budget under the clean selection protocol (`cvsel`,
§5.2). Retrieval is **zero-shot** cosine ranking against all 2,020 game vectors.
Tags are read by a ridge probe fitted on training-fold games only and scored on
fully held-out test games (`test_tag`). Fixed-split numbers are single-split and
carry ±0.03–0.04 checkpoint noise; they are labelled as such.

---

## 1. The data shape this model is for

An entity (a game) is observed through thousands of **unordered, mutually
discontinuous** short texts (player reviews). Two properties break the standard
self-supervised toolkit:

- **No token continuity between observations.** Review *i* and review *j* of the
  same game share no narrative, so masking / next-token objectives have nothing
  to span.
- **The unit of observation is not the unit of interest.** A single review is not
  the learning target; the game representation is the *consensus* of thousands.

**Corpus.** 2,020 games, 6.65M reviews, 73.29M sentences, embedded once by a
frozen Qwen3-Embedding-0.6B (1024-d) and never touched as raw text again. Games
enter only if ≥500 reviews of ≥300 characters survive filtering — that rule
reduces the 23,107-game source to 2,020.

---

## 2. Architecture — `SetPoolN`, 361,856 trainable parameters

```
S  : [B, L, 1024]   row-normalized sentence embeddings (per-row mean 0 / std 1, fp16)
m  : [B, L]         padding mask

q0   = nn.Parameter(randn(1, 4, 128) * 0.02)                        #     512
attn = MultiheadAttention(embed=128, heads=4, kdim=1024, vdim=1024) # 295,424
head = Linear(128→256) → GELU → Linear(256→128)                     #  65,920

a = attn(query=q0.expand(B), key=S, value=S, key_padding_mask=m)
z = normalize( head( a.mean(dim=1) ) )        # 128-d unit vector, one per game
```

Design consequences:

- **Cost is `Q·W`, not `W²`.** Four latent queries attend over a W-sentence set,
  so a 4,096-sentence anchor pack costs 4×4,096 interactions instead of 4,096².
  This is the only reason a *large* teacher view is affordable at all.
- **The asymmetric `kdim/vdim = 1024` vs `embed = 128`** is what keeps the model
  at 0.36M parameters while consuming full 1024-d sentence sets: keys and values
  are projected from raw embedding space directly into the 128-d working space.
- **One shallow attention layer — no residual, no LayerNorm, no FFN, no depth.**
  The tower is a pooling operator, not an encoder stack.
- **One vector per game.** Slot-mean then head; the output is a single 128-d
  vector, so nothing downstream is left unpooled.
- **One network for everything.** The same instance encodes anchors, all four
  student views, document views and evaluation queries. There is no projector,
  no separate teacher network.

**Slot readout is parameter-free by evidence, not by taste** — see §10.3.

---

## 3. Views and anchors

### 3.1 Student views (4 per game per step)

Views are drawn by **rejection sampling over whole reviews**. Per game, each
review of length L gets an acceptance probability

```
a(L) = 0.2 + 0.7 · (L − L_min) / (L_max − L_min)          # ∈ [0.2, 0.9]
```

(if all reviews have equal length, a = 0.9 everywhere). Sampling is **without
replacement** within a view: draw a random unused review, accept with
probability a, add it **whole**, repeat until the cumulative sentence count
crosses W.

**The prior encoded here:** long reviews tend to be the valuable ones — they
carry the story and mechanics a game representation needs, while short reviews
are disproportionately pure verdict ("great game, 10/10"). The 0.2 floor keeps
the short majority represented so the tilt does not become a filter; the 0.9
ceiling means even the longest review is not auto-included.

**W = 16 is a stopping threshold, not a cap.** The loop exits *after* the accept
that crosses W, and no review is ever split, so the realized view overshoots.
Measured over 2,020 games × 20 views: median **2 whole reviews** (mean 2.2),
median **23 sentences** (mean 29); one review fills the whole view 24.9% of the
time.

**The four views** are three independent review draws plus one **document
view**, assembled by tier: (1) a wiki-derived article if the game has one, (2)
the store page otherwise, (3) if neither exists, the game simply gets a **fourth
review view**. Tier dictionaries are built with the fold's exclusion set applied,
so no held-out game's document is ever encoded during training.

Masks are re-drawn fresh every step — see the fresh-mask law in §11.

### 3.2 Teacher anchors (the "pack")

Each game has one anchor pack: the store-page sentences as a **prefix**, then
whole reviews drawn in a random permutation and accepted only if the *entire*
review fits the budget (512 / 1,024 / 2,048 / 4,096 sentences). A review that
would overflow is **skipped**, and the scan continues so shorter later reviews
still fill the tail.

Two properties carry the whole design:

- **The pack is fixed** (a content pack, not a cache): the same sentences every
  step under a fixed seed. What varies across steps is the *student* view, never
  the target.
- **The pack is re-encoded with gradient at every step.** `gallery_train(model)`
  walks all training-pool anchors (1,694 five-fold / 1,613 fixed-split) in
  chunks through the live tower inside the autocast region, with no
  `torch.no_grad`. Every optimizer step backprops through the full gallery. (The `gallery` / `gallery_nodoc` variants are
  `no_grad` evaluation paths only.)

This is the expensive part of the recipe, and every economy arm in §10.4 exists
to attack it.

---

## 4. The objective

```
L = Σ_v  ℓ_CE(z^v)  +  λ · (1/6) · Σ_{u<v} ( 1 − cos(z^u, z^v) )
```

with λ = **2** for I-CE (λ = 1 is the weaker `ice` variant), and

```
ℓ_CE(z) = − log [ exp(z·a_g / τ) / Σ_{h=1..N_train} exp(z·a_h / τ) ],   a_h = f_θ(P_h)
```

- **CE term — identity.** Every student view is classified against the **full
  gallery of training-pool anchors**, all re-encoded with gradient — **1,694 at
  five-fold** (2,020 minus the 326 held-out games of that fold) and **1,613 on
  the fixed split** (2,020 minus 407). Positive: the view's own game anchor.
  Negatives: literally every other game — "batch = all", not a batch subset and
  not a queue. τ = **0.02**, frozen.
- **I term — structure.** Mean `1 − cos` over all **six pairs** of the four
  views, applied **per view and never on a pooled vector** (§10.3 shows what
  happens when you pool).
- The document view participates in both terms exactly like a review view.

### 4.1 Why no stop-gradient, momentum encoder, or predictor — and the evidence

Negative-free methods need those devices to prevent collapse. Here the CE term
supplies a **static energy barrier**: the constant solution costs
`ln 2020 ≈ 7.6` nats per view, so collapse is not a fixed point and no
stabilizing machinery is needed.

The stronger claim — that the teacher *must* receive gradient — was tested
directly and is the sharpest ablation in the campaign:

| Gallery treatment | Stripped hit@1 (fixed split) |
|---|---|
| Full stop-gradient (`i2sgce`) | **0.152** |
| Half stop-gradient | 0.574 |
| Fully learnable (the recipe) | **0.686** |

**Anchors must be learnable.** A detached teacher is not a cheaper I-CE; it is a
different, far worse model. Anchors and views co-adapt in both directions, and
that co-adaptation is what buys name-stripped retrieval.

The price of the choice is the memory ceiling: full-gradient re-encoding is
linear in `entities × pack size` (§9).

### 4.2 Why λ = 2 (the I-dose ladder)

| Invariance dose | Stripped hit@1 |
|---|---|
| I = 0 (plain CE) | 0.387 |
| I = 1 | 0.435 |
| I = 2 (the recipe) | 0.463 |

Monotone. The view–view term is the only free lunch found in the entire loss
ladder: adding the anchor itself to the I set (`ai2ce`) buys **+0.003 = noise**,
because CE's softmax pull on the anchor edge is already `(1−p⁺)`-adaptive and
saturates it. **The I term's value is view↔view, not view↔anchor.**

### 4.3 What the I term actually does — the mediating variable, measured

The obvious objection is that I should be redundant: CE already pulls all four
views toward the same anchor, so they must end up near each other anyway. It is
not redundant, and the reason is a **stopping condition**. CE's gradient is
proportional to `(1 − p⁺)`: once a view classifies correctly, the force
vanishes, so the four views may settle anywhere inside their game's cell. I has
no such stop — it compresses unconditionally.

Measured directly (fold-0 towers, 600 games, four disjoint random review
subsets per game drawn from the anchor pack's review region; `view_cos` = mean
pairwise cosine among a game's four view encodings, `nn_gap` = cosine to own
full-pack encoding minus cosine to the nearest *other* game's):

| Tower | view_cos | view → own game | nn_gap | Stripped hit@1 |
|---|---|---|---|---|
| CE @ep650 | 0.597 | 0.770 | +0.100 | — |
| CE @ep1450 | 0.582 | 0.757 | +0.111 | — |
| CE @ep1850 | 0.585 | 0.760 | +0.112 | 0.658 (5-fold) |
| I-CE @ep650 | **0.736** | 0.844 | +0.078 | — |
| I-CE @ep1450 | **0.739** | 0.842 | +0.085 | 0.732 (5-fold) |
| SimCLR @ep1850 | 0.857 | 0.908 | +0.052 | 0.624 |
| VICReg @ep1100 | 0.765 | 0.862 | +0.039 | 0.426 |
| BYOL @ep300 | 0.817 | 0.895 | **−0.020** | 0.278 |

Three readings:

1. **The "I-CE is just trained longer" confound is dead.** CE's consistency is a
   flat line across ep650 → ep1850 (0.597 / 0.582 / 0.585) — nearly three times
   the training changes nothing. I-CE is flat too, at a level 0.15 higher, from
   its earliest checkpoint. Epoch-matched at ep1450 the gap is **+0.157, higher
   in 100% of the 600 games**. Consistency is set by the loss, not by budget.
2. **But consistency is not monotonically good.** SimCLR has the highest
   `view_cos` in the table and retrieves 0.108 below I-CE. What orders retrieval
   is the *pair*: CE sits at high discriminability / low consistency, the
   negative-free family at high consistency / collapsed discriminability
   (BYOL's `nn_gap` is **negative** — its views are closer to some other game
   than to their own, which is identity collapse measured directly), and I-CE is
   the only row healthy on both.
3. **This is why I helps CE and cannot replace it.** I moves the encoder along
   the consistency axis without buying discriminability; CE buys
   discriminability and stops as soon as it has it. The geometry table (§11.5)
   shows the same split from the other side: adding I cuts query displacement by
   49% while leaving the nearest-neighbour margin essentially alone.

---

## 5. Training and selection

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr 5e-4, weight decay 1e-4 |
| Batch | 192 games/step, 16 steps/epoch, **2,000 epochs** |
| Temperature | τ = 0.02, frozen |
| Checkpoints | every 50 epochs |
| Precision | AMP, grad-clip 5.0 |

### 5.1 Why the budget stops at 2,000 epochs

Extending I-CE@4096 to 4,000 epochs produced a **catastrophic late collapse**,
not further gains: stripped hit@1 drifts 0.68 → 0.62 from ~ep2800, spikes
erratically (0.42 / 0.61 / 0.50), then free-falls past ep3300 to **0.02–0.10** at
ep4000; neutral falls 0.94 → 0.05. The cause is a late loss-spike instability
under a constant lr with a sharp τ, not slow overfitting. Validation signals
crash *in sync*, so validation-based selection avoids the cliff — but the budget
stands at 2,000.

### 5.2 Selection (`cvsel`) — fully inductive and rewrite-free

```
cvsel = stripped_hit@1(val) + stripped_hit@5(val) + 2 · val_tag
```

`val_tag` fits the ridge probe on **training-fold games only**, thresholds on the
validation fold, and scores the validation fold. Test games never touch
selection. The tag column reported everywhere (`test_tag`) applies the identical
probe to held-out test games: **validation selects, test only reports.**

Switching from the earlier leaky selector to `cvsel` moved the selected
checkpoint in **37 of 40** fold-selections — selection rule changes are not
cosmetic at this noise level.

### 5.3 Evaluation is zero-shot by decree

A fine-tuned readout head was tested and **damages every tower**: −0.008 for the
most robust variant, up to −0.112/−0.139 for pure CE. The cause is a query-type
mismatch (the head trains on review pseudo-queries; evaluation queries are
articles). Even a selection-free zero-shot checkpoint beats the head everywhere,
so the head was dropped from the protocol entirely. One consequence: an apparent
"noname weakness" of I-CE@4096 was a **head artifact** — at zero-shot its
stripped retrieval was the best on the board.

---

## 6. Results (five folds, @4096)

| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set TAG F1 |
|---|---|---|---|---|---|
| **I-CE (ours)** | **0.947 ± 0.021** | **0.996 ± 0.005** | **0.732 ± 0.013** | **0.923 ± 0.024** | 0.698 ± 0.011 |
| CE (contrast only) | 0.931 ± 0.015 | 0.990 ± 0.011 | 0.658 ± 0.016 | 0.883 ± 0.014 | 0.676 ± 0.028 |
| SimCLR-style (in-batch) | 0.919 ± 0.026 | 0.989 ± 0.005 | 0.624 ± 0.027 | 0.881 ± 0.028 | 0.711 ± 0.018 |
| VICReg (epd 20/10/20) | 0.775 ± 0.041 | 0.939 ± 0.009 | 0.426 ± 0.034 | 0.715 ± 0.034 | 0.713 ± 0.029 |
| BYOL | 0.441 ± 0.032 | 0.731 ± 0.029 | 0.278 ± 0.055 | 0.547 ± 0.038 | **0.717 ± 0.020** |
| Frozen embedder (no tower) | 0.381 ± 0.039 | 0.598 ± 0.021 | 0.167 ± 0.043 | 0.346 ± 0.023 | 0.592 ± 0.012 |

**The factorization.** Contrast buys identity; negative-free alignment buys
semantics; I-CE inherits both. Adding I to CE lifts every column at once
(stripped 0.658 → 0.732, +0.074, 5/5 paired folds) *and* raises the tag reading
(0.676 → 0.698) rather than paying for it.

**Anchor-budget scaling** (I-CE stripped hit@1): 0.644 @512 → 0.705 @1,024 →
0.701 @2,048 → 0.732 @4,096. The steep gain is up to 1,024 — the cost-effective
operating point, within 0.03 of the largest budget at a quarter of its memory.
I-CE leads CE at every budget (18 of 20 paired folds). Scaling is **not
monotone** (the @2,048 rung sits flat), so the claim is saturation, not a law.

> **Scope of the ladder.** The budget sweep exists only for **CE and I-CE**,
> because only their anchor packs enter the loss. `bce` (SimCLR-style),
> `byol` and `epd` (VICReg) never touch the anchor gallery during training —
> verified in code: `train_byol` and `train_vicreg` contain zero gallery
> references, and the `BCE` branch takes its logits from the step's own
> `4 × 192` view encodings. For those arms `--anchor-cap` only rebuilds the
> **evaluation** gallery, so a "@1024 BYOL" would be the same trained model
> read against a smaller index, not a different model. That is why all three
> baselines exist at @4096 only, and why the cost-effective-budget finding is
> a statement about **I-CE's own teacher budget**, not a cross-method
> recommendation.

---

## 7. The temperature dial: τ = 0.02 costs TAG

**τ is the single knob that moves capability between identity and semantics, and
τ = 0.02 deliberately sacrifices tag F1 to buy retrieval.**

| τ | Budget | Stripped hit@1 | Test-set TAG F1 | vs τ = 0.02 |
|---|---|---|---|---|
| **0.02 (recipe)** | 512 | **0.644** | 0.698 | — |
| 0.05 | 512 | 0.588 | 0.711 | tag **+0.013**, retrieval **−0.056** |
| 0.10 | 512 (4 folds) | 0.497 | 0.712 | tag **+0.014**, retrieval **−0.147** |
| **0.02 (recipe)** | 2,048 | **0.701** | 0.697 | — |
| 0.05 | 2,048 | 0.647 | 0.713 | tag **+0.016**, retrieval **−0.054** |
| 0.10 | 2,048 | 0.494 | 0.706 | tag **+0.009**, retrieval **−0.207** |
| learnable (init 0.02) | 2,048 | 0.716 | 0.691 | tag **−0.006**, retrieval **+0.015** |

**Mechanism.** A sharp temperature concentrates the softmax gradient on the few
*most confusable* neighbours — the games nearest in embedding space. Learning is
spent separating individuals, which is exactly identity. A soft temperature
spreads gradient across all negatives, so the model is graded on coarse
structure instead, which is exactly what a tag probe reads. The two readings
move in opposite directions under τ, and no setting we tested buys both.

**The learnable-τ run is the cleanest evidence that this is a real frontier and
not a tuning oversight:** allowed to choose, the temperature drifts to the
**identity-first end** — it gains stripped retrieval (+0.015) and gives up tag
(−0.006) against frozen 0.02. Nothing in the objective wants the tag end.

**Verdict.** Tag and name capability cannot be maximized simultaneously in one
128-d vector under this loss family. τ = 0.02 is our declared position: maximum
identity, tag F1 within ~0.02 of the best alternative.

---

## 8. Where I-CE loses: the TAG task

I-CE does **not** win the tag reading. Five models beat it, all lighter on
negatives. Paired per-fold comparison (same folds, same splits, same probe):

| Model beating I-CE on TAG | TAG F1 | Δ TAG | TAG wins | Δ Stripped hit@1 | Retrieval wins |
|---|---|---|---|---|---|
| **BYOL** | 0.717 ± 0.020 | **+0.019** | **5/5** | **−0.455** | 0/5 |
| **VICReg (epd 20/10/20)** | 0.713 ± 0.029 | **+0.015** | 3/5 | **−0.306** | 0/5 |
| **EMA + memory bank (I-CE)** | 0.712 ± 0.014 | **+0.014** | 4/5 | **−0.124** | 0/5 |
| **SimCLR-style (in-batch)** | 0.711 ± 0.018 | **+0.014** | 4/5 | **−0.108** | 0/5 |
| **Two-stage (BYOL warm → swin)** | 0.709 ± 0.017 | **+0.011** | 4/5 | **−0.049** | 0/5 |
| *I-CE (ours)* | *0.698 ± 0.011* | — | — | — | — |
| swin-I-CE (windowed) | 0.697 ± 0.009 | −0.001 | 2/5 | −0.015 | 0/5 |
| CE (contrast only) | 0.676 ± 0.028 | −0.021 | 1/5 | −0.074 | 0/5 |

1. **The gaps are small; the prices are not.** The largest tag deficit, 0.019,
   costs BYOL 0.455 stripped hit@1 — it cannot resolve identity at all. The
   smallest, 0.011, comes from the only model that also retrieves respectably.
2. **Not one of them wins a single fold on retrieval — 0/5 across the board.**
3. **The semantic band.** Every negative-light objective converges to the same
   narrow tag band, **0.711–0.717**, regardless of mechanism (in-batch views,
   memory bank, momentum target, variance regularization). I-CE reads
   0.013–0.019 below it. The band behaves like a data-imposed ceiling on what a
   tag probe can extract; what differs between methods is only how much
   retrieval they pay to sit in it (0.05–0.46).

---

## 9. Economies of anchor supply

| Variant | Mechanism | Stripped hit@1 | vs full coupling | Test TAG |
|---|---|---|---|---|
| I-CE, fully coupled | whole gallery, gradient, every step | 0.732 | — | 0.698 |
| swin-I-CE (W=168, S=84) | ring window, ~27% of gallery with gradient per step, ~40% less memory | 0.717 | **−0.015** | 0.697 |
| Two-stage (BYOL → swin, 600 ep) | negative-free warm start, then windowed calibration at 30% of the budget | 0.683 | **−0.049** | 0.709 |
| EMA + memory bank (3,072 keys) | teacher replaced by EMA shadow encoder feeding a bank | 0.608 | **−0.124** | 0.712 |

**Law: how the budget is saved matters more than how much.** The window and the
bank save comparable memory; the window pays 0.015, the bank 0.124. §10.4
explains why.

---

## 10. Complete arm registry and failure modes

### 10.1 Total collapse — retrieval hit@1 ≈ 0

| Arm family | Mechanism | Result |
|---|---|---|
| `ai2auni25`, `ai4auni2`, `ai6auni2` + 13 gated cells | **anchor-field uniformity**: replace the CE softmax with an explicit align (view→own anchor) + log-partition/Gaussian uniformity push against wrong anchors | **hit@1 0.000 / 0.009 / 0.000**; every embedding collapses to one point (hit@5 = 1/204); 13/13 gated variants died too |
| `cmpai25*auni2`, `expai25*auni2`, `pjai25*auni2` (8 arms) | same, under compare / expand / project pooling heads | **hit@1 0.000**, tag 0.15–0.42 |
| `vic` / `vic2` (centroid wiring) | VICReg invariance as MSE between unit-norm **centroids**, V/C on `expander(centroid)` | **hit@1 ≤ 0.03** at every view width; exact centroid collapse zeroes the MSE while V/C stay satisfied — a legal degenerate optimum |
| `epda` (decoupled VICReg) | invariance on views, V/C on gallery population moments | instant collapse; ruled a misdesign and purged |
| pure DINO self-distillation | full teacher distillation without anchor CE | name recall **0.065** |

**Failure mode — the satisfiability asymmetry.** A log-partition uniformity push
is *scale-invariant*: it exerts constant total gradient at any distance and is
never satisfied. A constant-weight cosine attraction rope is bounded and gets
*weaker* as points approach. There is therefore **no zero-force fixed point** —
the field drifts forever until every embedding piles onto one pole. The anchors
also all live inside a single text-embedding cone (anisotropy), so wrong-anchor
pushes never cancel. A softmax has neither problem: its push is
`(1−p)`-weighted, so it *does* have an equilibrium.

The lesson is asymmetric: **the I term cannot substitute for CE, but CE can
carry I.** For the same reason, adding MV-InfoNCE to an anchored CE was inert —
it treats view–view disagreement that the shared anchor prototype has already
cured.

### 10.2 Anchor memory banks and caches — the key-space consistency law

| Arm | Mechanism | Result |
|---|---|---|
| `bkq192` / `bkq48` / `bkq12` | one 128-d snapshot row per entity, k rows re-encoded per step cyclically, **no gradient through any anchor** | **collapsed at every refresh rate**: 0.410 / 0.406 / 0.413 — even an 8.4-step full-refresh cycle |
| `bkb` | refresh = the current batch, so positives are always fresh *and* carry gradient while negatives are stale snapshots | **0.267 — the worst of all**, best_ep 50, decays 0.265 → 0.19 |

**Law: the binding constraint is key-space consistency, not freshness.** Refresh
rate is irrelevant when keys come from many different past students. Worse,
`bkb` shows that *mixing* time-scales is actively harmful: if anything current
beats anything old, the tower can win CE by **drifting** instead of learning
identity. This is exactly why MoCo needs a single momentum encoder — and it is
why the windowed teacher was designed with **no cache anywhere**.

`mq3072` (MoCo-style EMA shadow + 3,072-key bank) is the disciplined version and
survives, but pays 0.124 stripped hit@1 five-fold (0/5 folds), with the entire
cost landing on the name-stripped column.

### 10.3 Readout and loss-attachment — pooling is a shortcut

| Family | Mechanism | Result |
|---|---|---|
| pooled CE (16 structure cells) | CE on the normalized **mean** of the views | **16/16 lose** to their per-view siblings on every axis: pooled 0.572 neutral / 0.261 stripped vs per-view 0.877 / 0.575 |
| `i2poolce` | pooled CE with I retained | a **late cliff**, not a decline: stripped 0.475@ep200 → collapse at ~ep550 → 0.083@ep1000 |
| expander / compressor grid (30 cells) | CE and/or I moved into a projected space (128→256→512 or 128→128→64), shared vs dual heads | **every attachment costs 0.035–0.064** stripped; the structure optimum is plain deployed per-view CE + deployed I |
| `slot{4,8,16}line` | learned `Linear(N·128→128)` slot pooling, initialized exactly to mean-pool | catastrophic and worsening with N: 0.162 (slot8), 0.054 (slot16) vs 0.662 / 0.657 for mean-pool |
| output-side per-source heads (`pi2ccec`, `qpi2cce`) | I/C behind per-source (review/wiki/store) projection heads | crashed (identity dead by ep50, NaN in the tag ridge); the deployed space was left with nothing anchoring `‖h−μ‖` |

**Failure mode — encoder-policy space.** Pooling before the loss constrains only
the *mean direction*, so the encoder learns an input-conditional policy: push
signal into the easy views, cancel noise in the rest. Per-view evaluation then
collapses. (A naive linearity argument suggests pooled ≡ per-view; that argument
is wrong once the encoder can choose *which* view carries the signal.) **CE must
be per-view** is now a hard rule in the worker.

### 10.4 Targeted / sparse anchor supply — the death map

| Arm | Mechanism | Result |
|---|---|---|
| `spb64i2ce` | full no-grad gallery for the CE partition, but gradient only through positives + per-view **top-64 loss-mass** anchors (≈25% activation) | **dead twice**: from scratch 0.108 stripped, pinned at neu ~0.30 / non ~0.13 for 2,000 epochs with byte-identical logs after ep800 (representation frozen) — *worse than the frozen-embedder baseline*; BYOL-warm 0.324 |
| `nm16` (nemesis) | each game's top-K nearest anchors occupy a **fixed block** of the swin window; index refreshed every 50 epochs | negative: 0.662 (−0.029), no tag gain — stale index + fixed blocks crowding out random-negative diversity |
| `swin{84,168,336}` | sliding fresh window over a ring, two half-overlapping micro-passes, immediate backward, **no cache** | **lives**: 0.717 five-fold (−0.015). Window size is a plateau: 19.7% / 27.5% / 43.1% coverage all within single-split noise |

**The anchor-supply death map.** `spb` gets gradient to 25% of the gallery and
dies; `swin` gets 27% and lives. The difference is not *how much* coverage but
whether coverage is **unbiased in time**. The window rotates deterministically —
every anchor takes a regular turn. Sparse backward re-selects by *current loss*,
so the same "easy" anchors are starved indefinitely. The 97%-gradient-mass
measurement that justified top-64 was taken on a **converged** tower; early in
training the softmax is nearly flat and top-64 covers almost nothing.

**Corollary: gradient-mass concentration ≠ functional importance.**

### 10.5 Negative-free families

| Arm | Mechanism | Five-fold | Failure mode |
|---|---|---|---|
| `byol` | online + EMA target (m=0.996) + predictor, `1 − cos` | stripped **0.278**, tag 0.717 | **Identity collapse, semantics intact.** No repulsive force ties a game to its own identity; NN margin 0.050 vs I-CE's 0.170, queries overshoot threefold. UMAP shows filamentary identity collapse. |
| `byol2` | + BatchNorm in projector and 3-layer BN predictor | probe 0.340 → **0.521** | BN's cross-sample statistics are the implicit-contrast channel BYOL otherwise lacks — the one real rescue; also the only family that gains monotonically from wider W. Label smoothing on top, by contrast, hurts. |
| `mv3` | MoCo-v3 analog: symmetrized **cross-entropy** (per the original paper, not cosine) + EMA target + predictor | — | built to the paper's symmetric CE form on user instruction |
| `epd_v20i10c20` | VICReg on expander outputs over six view pairs, 192 games/step | stripped **0.426**, tag 0.713 | **"Room without aim."** The variance term manufactures distance as a *batch statistic* that never couples queries to anchors; top-10 neighbourhood purity equals the frozen baseline. |
| `epdb_*` | same, **batch = all** (views for every training game every step) | fold-1 0.387 | ~8× step cost (22 h/fold) for nothing; also OOM-wiped five folds until anchors were parked in pinned host RAM. Retired. |
| `epd_v25i25c1`, `_v20i10c15`, `epdg_*` | VICReg weight sweep | — | image-domain 25/25/1 is wrong for unit-norm 128-d: 0.326 vs 0.449 for 20/10/20 |
| `i4uni2`, `i6uni2` | pure Wang & Isola on **views only** (align 2 or 3 + Gaussian uniformity), anchor-free | stripped 0.186 / 0.211 | Batch self-repulsion self-isotropizes, so no pole collapse — but uniformity grinds away the doc↔review coupling. Peak at ep50–150 then decay. **Tags stay intact (0.721 / 0.728)**; `i4uni2` is even the tag_neutral champion at 0.746. |

### 10.6 Teacher-construction experiments

| Arm | Mechanism | Result |
|---|---|---|
| `pk2i2cevi2ce` | **twin teacher packs**: N packs per game, each an *independent random sample* of the game's anchor sentences (packs **may overlap** — disjoint contiguous slices were explicitly rejected), fixed per-game seed. The pack level runs its own full I-CE: **pack-CE** (each pack classified against the alive-weighted mean gallery) + **pack-I** (`1 − cos` between every alive pack pair, weight 2.0), *in parallel with* the ordinary view-level I-CE. Deploy gallery = normalized mean of the packs. The doc prefix is pinned to pack 0 only; doc-only games keep pack 0 alive and mask the rest. | **not a confirmed win** — see the audit below |
| `pk2i2vce` / `pk2i2cevce` / `pk2i2cesgvce` | ablations of the above (no pack-CE / no view-I / detached mean) | all negative: 0.603 / 0.647 / 0.574 — *if* twin packs are used, pack-CE and view-I are both load-bearing |
| `pk4i2cevi2ce` | four packs at the 2,048 budget | 0.676 — **exactly equal** to I-CE@2048's 0.676, with lower neutral (0.931 vs 0.946) |

**Audit of the twin-pack claim.** The headline reading was `pk2i2cevi2ce`
@1024 = 0.716 against I-CE@1024 = 0.657, and it does not survive scrutiny:

1. **The two numbers come from different selection rules.** The pk checkpoint was
   picked by `rvsel` (ep1900); the I-CE@1024 baseline carries only `zvsel`
   (ep1100). This is not a same-protocol comparison.
2. **The mechanism's own scaling test is flat.** If splitting the budget into
   packs beat one joint attention pool, four packs at 2,048 should benefit too.
   `pk4i2cevi2ce` ties its baseline to three decimals (0.676 vs 0.676).
3. **The clean control erases the gap.** Under five-fold `cvsel`, plain
   I-CE@1024 reads **0.705 ± 0.030** (folds 0.706 / 0.681 / **0.755** / 0.669 /
   **0.716**). The pk single point of 0.716 sits +0.36 σ from that mean — and two
   individual I-CE folds match or beat it.

Twin packs were never run at five folds and have no paired statistics. The
honest status is **unconfirmed**, not "the one construction that beat joint
pooling". Its +0.059 is fully explained by a selection-rule difference plus
single-split checkpoint lottery.
| `vfai2ce` | **views-first anchors**: the step's four student views enter the teacher pack first, the fixed pack fills the rest | **double loss**: neutral 0.901 → 0.638, stripped 0.657 → **0.398** (0/5), tag 0.705 → 0.670 (0/5), `best_ep = 50` in **every** fold |
| `i2sgce` / `i2q2ce` / `i2esce` | three attempts to stop late-training tag decay at large caps: stop-grad gallery / soft-band I / EMA-shadow gallery | `i2sgce` **fatal** (0.152); `i2q2ce` no-gain; `i2esce` a Pareto endpoint only (tag 0.745 bought with retrieval 0.618) |
| `nodoc` | zero document views — all four views are reviews | stripped 0.615, −0.028 vs 0.644 (4/5 folds) |
| `i2ce_wllm` | document views replaced by faithful sentence-wise **LLM rewrites** (contamination firewall) | 0.642, **−0.001** — the firewall is free at fold level |

**Positive-leakage law (`vfa`).** If the teacher pack contains the student's own
sentences, the CE positive becomes trivially matchable, the softmax stops forcing
identity, and at evaluation — where the teacher reverts to the fixed pack — the
student cannot recognize it. Monotone degradation from the first checkpoint, in
every fold. **The teacher must exclude the student's own view.**

### 10.7 Gating experiments (which games get which loss)

`cegate1/2/3/4`, `cegate1w/2w`, `igate1`, `igate1w` fire the CE (or I) term only
on document-bearing games; `rgate2` fires it on a **coverage-matched random**
set of the same size — the control that separates "document coverage" from mere
"CE dose reduction".

**CE-gating law, with a cliff.** Narrowing CE to the 1,415 document-bearing
games *lifts all four axes*; narrowing further to the 407 wiki-only games falls
off a cliff (0.60–0.62, peaking at ep50 = identity bleed from a starved negative
set). Train-time centering (`cegate2c`) was neutral-to-negative, and post-hoc
evaluation-side centering also lost (0.875 → 0.864) — both directions closed.

### 10.8 Distributed / scale prototypes

| Arm | Mechanism | Result |
|---|---|---|
| `as8dc5i2ce` | DC-ASGD simulation: 8 workers, delay compensation `g' = g + λ·g⊙g·(W_t − W_pull)`, λ = 0.5 | simulation arm |
| async PS v1 | real parameter server, no backpressure | **0.162** — queue divergence, staleness reached 2,473 versions, 96% of updates clamped |
| async PS v3/v4 | + backpressure, drop-stale, epoch-push, hard barrier, sharded overlapping data pools (cover 0.3), rotating borrow slices prefetched from a disk tier | fixed split stripped **0.603** vs 0.691 synchronous swin; tag 0.740. Prefetch hit 39/40 rotations, staleness 1.0, but `best_ep = 1950` — under-trained (2,000 optimizer steps vs 32,000) |

Six-plus crash rounds preceded a working PS, all root-caused and recorded:
`torch.save` rejecting a dot-prefixed basename, scope errors from verbatim
duplicate lines shared with another function, OOM from co-packing, and gallery
tensors parked on the master that it never reads during rounds.

---

## 11. Laws the campaign established

1. **Negatives buy identity; negative-free alignment buys semantics.** Same
   tower, same views, same budget — only the loss differs, and the two
   capabilities separate cleanly (SimCLR vs CE: identity 5/5 to CE, semantics
   5/5 to SimCLR).
2. **The semantic band.** All negative-light objectives converge to test-tag
   0.711–0.717 regardless of mechanism, at stripped-retrieval prices of
   0.05–0.46. I-CE sits 0.013–0.019 below it and pays none of them.
3. **The dial law — identity ↔ semantics trades at ~2:1.** Three structurally
   unrelated knobs (temperature softening, forced slot decorrelation, an EMA
   teacher) all price at roughly 2 retrieval for 1 tag. Same exchange rate from
   unrelated mechanisms ⇒ a real frontier, not a tuning failure.
4. **~30 effective dimensions / 93–95% readout redundancy.** The four attention
   slots are structurally distinct (pairwise `q0` cosines −0.19 to −0.33, block
   effective rank 2.85/3) but 93–95% redundant *at the readout*. Extra capacity
   is never paid for: slot8 vs slot4 is +0.016 in 3/5 folds = noise. **Capacity
   is flat**, and `Q1 ≈ Q4` when trained long enough.
5. **Retrieval is a ratio, not a distance.** `displacement / NN-margin` orders
   retrieval quality exactly. The frozen embedder has the *smallest* displacement
   (0.157) and the worst retrieval, because its margins are 0.012 (ratio 13.07).
   I-CE alone pushes the ratio toward one (1.18) — by buying margin faster than
   displacement grows. **CE fixes the denominator, I fixes the numerator.**
6. **Time-unbiased gradient coverage is the survival criterion** for any anchor
   economy (swin lives at 27%, spb dies at 25%).
7. **Key-space consistency, not freshness**, is what a cached teacher must
   preserve; mixing time-scales is worse than uniform staleness.
8. **Anchors must be learnable** (stop-grad 0.152 / half 0.574 / full 0.686).
9. **The teacher must exclude the student's own views** (positive leakage:
   0.657 → 0.398, 0/5).
10. **CE must be per-view; pooling before the loss is always a shortcut**
    (16/16 pooled cells lose).
11. **Fresh masks ≫ fixed subsets** (+0.10 neutral), and **wider student views
    monotonically hurt** (stripped 0.277 → 0.075 as W grows), with variance
    exploding at the top end.
12. **Never compare across selection rules.** Switching from the leaky selector
    to `cvsel` moved the chosen checkpoint in 37 of 40 fold-selections, and a
    single fixed split carries ±0.03–0.04 checkpoint lottery on top. Any
    cross-arm claim built on two differently-selected single points is
    unfalsifiable — the twin-pack "win" above is exactly that failure, and it
    dissolved once a same-protocol control was run.
13. **Contamination is total-radiation.** Raw wiki text is banned from training,
    fine-tuning *and* evaluation because the evaluation queries are wiki-derived;
    the legal substitute is a sentence-faithful LLM rewrite (audited: longest
    verbatim overlap with sources = 31 characters). Several record-holding arms
    were struck from the record under this rule.

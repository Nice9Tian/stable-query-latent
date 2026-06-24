# Tag prediction on real descriptions (AO_text / 2077_text)

Test: predict a game's non-emotional Steam tags from a real Chinese
mechanism/story description. Ground truth = the game's actual Steam tags.
Metric = precision@15 / recall@15 over the 189 non-emotional tags. Both games
(Across the Obelisk appid 1385380, Cyberpunk 2077 appid 1091500) are in the
training set, so we have true tags.

## What every approach scores (prec@15)

| approach | Across the Obelisk | Cyberpunk 2077 |
|---|---|---|
| **current deployed VICReg-stats probe** | **0.13** (saturated, all prob=1.0) | low |
| raw review-embedding probe, StandardScaler | 0.47 | 0.27 |
| raw review-embedding probe, L2-norm | 0.40 | 0.20 |
| description-trained probe, 293 games | 0.27 | 0.40 |
| description-trained probe, 3000 games | 0.07 | 0.40 |
| zero-shot tag-name ↔ sentence matching | 0.33 | 0.27 |

**Nothing reaches ~0.5, let alone 0.85.** The best result (raw review probe, 0.47)
already exceeds what the VICReg code can do, and the VICReg code is a function of
that embedding, so **no encoder retraining can beat the raw-embedding ceiling.**

## Two separate problems

1. **Saturation (a real bug, fixable).** The current deployed probe uses
   `StandardScaler` on review features; a Chinese description embeds out-of-domain,
   z-scores explode, and `sigmoid` saturates → "many tags at confidence 1.0".
   L2-normalizing features removes the saturation (frac prob>0.99 → 0.000). This is
   worth fixing regardless.

2. **The 85% target is not reachable on this task (fundamental).** Reasons, with
   evidence from the runs above:

   - **Tag redundancy.** AO's true tags include *Turn-Based, Turn-Based Strategy,
     Turn-Based Combat, Turn-Based Tactics* (4 near-duplicates) and *Card Game,
     Card Battler, Deckbuilding* (3 overlapping). 189-way exact prediction over a
     redundant, crowd-voted label set caps precision low — even a human can't pick
     which 4 of 6 "turn-based" variants the Steam crowd happened to vote.
   - **Crowd-label arbitrariness.** Steam tags are vote-ranked, capped at 20/game.
     Many mechanic tags a description clearly supports are simply not in the game's
     tag list (and vice-versa), so they count as "wrong" no matter the model.
   - **Mean-pool blur.** A long multi-topic description averages to a generic
     "game-ish" vector, so common tags (*Story Rich, Adventure, Anime, Female
     Protagonist*) dominate the top — visible in every probe variant.
   - **Domain + language shift.** Encoder/probe trained on English reviews; test is
     Chinese descriptions. Qwen is multilingual so it's partial, but the VICReg
     encoder (trained only on review-domain embeddings) produces a degenerate code
     on this input — that's the 0.13.

## What the model does get right

On AO the raw probe surfaces *Card Game, Deckbuilding, Card Battler, Turn-Based\*,
Co-op, Choices Matter* — it clearly captures the game's identity. The failure is
the exact 189-tag-match metric at 0.85, not "does it understand the game".

## Honest recommendation

- **Fixable now:** switch the deployed probe to L2-normalized features (kills the
  confidence=1.0 saturation; raw L2 probe is the best real-text performer). I can
  apply this to the export + validation.
- **To raise real accuracy meaningfully** (not to 0.85, but well above 0.13):
  evaluate on a **deduplicated coarse mechanic vocabulary** (collapse the
  turn-based / card / roguelike families) and/or use **per-sentence max-pooling**
  instead of mean-pool so a sentence like "co-op roguelite deckbuilder" votes
  directly for those tags.
- **What will NOT get to 0.85:** retraining the VICReg encoder (more layers /
  residuals / output dim). It is upper-bounded by the raw-embedding ceiling (~0.47
  here), which itself is far below 0.85 for the reasons above.

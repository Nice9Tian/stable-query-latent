# data/results/ — training artefacts

Written by the trainers (`steam_reviews_framework/run.py` for the
champion, `contrast_experiment/run.py` for the contrast suite); aggregated
by `contrast_experiment/report.py`. Naming conventions (`{arm}` = arm
name, e.g. `champion_cegate2`; CV arms = `cv_{recipe}_fold{k}`):

| File | Structure | Meaning |
|---|---|---|
| `ckpt_{arm}_ep{K}.pt` | torch state_dict | tower checkpoint, one every 50 ep (1,000 ep, no early stop) |
| `tower_{arm}_ep{K}.npz` | `SPg` (2020,128) anchor projections; `SPg_nd` reviews-only-anchor projections; `SPa` eval-query projections; `SPq` pseudo-query projections; `SPd` + `SPd_gidx` family-doc projections | ALL projections of the frozen tower at that checkpoint — the head stage consumes only this file and never touches the tower again |
| `zs_traj_{arm}.json` | `{ep50: {nm_neutral, nm_noname, tag_neutral, tag_noname}, …}` | bare-tower zero-shot trajectory across checkpoints |
| `ft4var_{arm}_ep{K}{head}.json` | `{per_seed: [{neutral:{h1,h5,med,tag}, noname:…, positive:…, negative:…, vscore, v_neu, v_non, v_non5}, …]}` | head results at that checkpoint (pass 1 = 3 seeds); `{head}` empty = main head, `_p1ce` / `_ce` / `_cetf` = ablation heads |
| `ft4var_{arm}_best{head}.json` | as above + `best_ep` + `vsel_by_ep` | **the post-hoc pick**: best checkpoint by mean vsel (validation only, never test), topped up to 10 seeds |
| `summary.md` | markdown table | report.py aggregate: fixed-split ranking + CV mean ± std |

Metric glossary: `h1`/`h5` = hit@1 / hit@5 against the full 2,020-game
gallery, `med` = median rank, `tag` = 23-tag micro-F1, `m4` = mean hit@1
over the four variants; `vscore` = the vsel selection score (best of the
two noname axes + neutral as an additive sanity gate — see
`steam_reviews_framework/protocol.py`).

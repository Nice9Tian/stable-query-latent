# larice — Latent Represent I-CE

A small (~0.4 M params), task-agnostic set-representation tower
(architecture lineage: SetPoolN) that turns a *set* of upstream
embeddings (sentences, patches, events, …) into one invariant vector.
`N` learnable latent queries cross-attend over the set; the training
objective factorises into two orthogonal forces:

- **CE (data axis)** — an InfoNCE pull toward the item's *anchor* in a
  gallery of all items, buying **identity**;
- **I (view axis)** — cosine alignment across independently sampled views
  of the same item, buying **semantic stability**.

The champion recipe gates CE (it fires only on items that carry a
document-grade view) while I always fires — negatives buy identity,
alignment buys meaning.

## Tensor protocol

The leading two axes of every input are **`[data, view]`**:

```
x    : [B, V, S, D_in]   float   S = set elements, D_in = upstream dim
mask : [B, V, S]         bool    True = padding
out  : [B, V, out_dim]           L2-normalised
```

Single-view tasks use `V = 1` (a rank-3 `[B, S, D_in]` input is accepted
and treated as `V = 1`). Loss semantics follow the axes: `invariance_loss`
reduces along **view**, `gated_ce_loss` reduces along **data**. Any task
whose samples can be laid out this way can use the tower unchanged.

## Readout

Default output is the **concat** of the `N` query slots
(`out_dim = N × dim_model`) — the general-purpose representation; each
slot is a deployable sub-space of its own.

> **Note — name/identity recall:** for retrieval-style tasks where the
> query must hit one specific item (name recall), the **pooled** readout
> (`readout="pool"`: mean over slots → MLP → L2) consistently outperforms
> concat in our experiments. Use `LariceConfig(readout="pool")` there.

## Usage

```python
from main_model import LariceConfig, LariceTower, champion_loss

cfg = LariceConfig(num_queries=4, dim_model=128, input_dim=1024,
                  num_views=4, tau=0.02, inv_weight=2.0, readout="pool")
tower = LariceTower(cfg)

z = tower(x, mask)                      # [B, V, out_dim]
loss = champion_loss(z, gallery, targets, cfg, gate=has_doc_view)
```

`gallery` is the tower's own encoding of every item's anchor set
(recomputed each step, gradients on — "batch = all" negatives). A fixed,
medium-cardinality anchor per item removes the need for EMA targets,
negative mining, or memory queues.

Config surface: `num_queries (N)`, `num_views (NV)`, `tau`
(frozen / learnable), `inv_weight` + CE gate, `dim_model`, `num_heads`,
`input_dim`, `readout`.

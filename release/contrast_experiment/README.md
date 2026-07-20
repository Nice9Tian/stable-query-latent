# contrast_experiment — the full contrast suite

Trains **every tower except the champion** and produces the comparison
(the champion is path 1: `steam_reviews_framework/run.py`). The
declarative arm roster lives in `contrast_models/roster.py` (18 arms:
I-CE family / pure CE / ArcFace / BYOL / the CE-gate I-dose ladder / the
I-gate mirror / random-gate and nodoc controls / wiki_llm leak ablations);
matching head grids are in `contrast_heads/configs.py`.

```bash
python contrast_experiment/run.py                 # one-click: data prep + 18 arms + table
python contrast_experiment/run.py --arms ce byol  # a subset
python contrast_experiment/run.py --cv            # + 6 recipes × 5 folds CV
python contrast_experiment/report.py              # regenerate the summary table
```

Everything is resume-safe: finished towers / heads / folds are skipped on
relaunch, so the run can be interrupted at any point.

## pod/ — the multi-machine parallel route (this layer's accelerator)

**The full contrast suite is a large set of mutually independent jobs
(18 arms + 30 CV fold-runs) — a natural fit for multi-machine
parallelism.** `pod/` is a complete recipe for spraying the whole roster
across N machines on RunPod (or any GPU cloud with a shared network
volume):

- **Distributed storage shares the data**: every machine mounts the same
  network volume (equivalent to an S3 bucket). Assets are prepared ONCE
  (atomic rename + READY marker, resumable), and results converge on the
  volume — any single machine can emit the complete comparison table;
- **Atomic multi-machine claims**: each job is claimed by
  exclusive-creating a claim file, so an arm is only ever trained by one
  machine; if a machine dies, its claim expires after 12 h and another
  machine takes over — **N machines ≈ N× throughput, no collisions, no
  central scheduler**;
- **Local staging**: `h5_staging.parallel_copy` bulk-copies the full
  review pool to local disk / shared memory before mmap-ing it (random
  reads on a network volume are slow; this step is an order-of-magnitude
  speedup);
- **Resume**: workers persist a rolling resume bundle at every checkpoint
  and continue from it after a restart;
- **Self-stop**: when the queue is empty and the audit passes,
  `pod_selfstop` shuts the pod down — no idle burn.

Usage: open one copy of `pod/w9_all.ipynb` per machine and run it top to
bottom. The cheapest pattern is to let the FIRST machine finish the
"ONE-TIME asset preparation" cell before starting the others. The job
list sits in the notebook's first cell (the same arm definitions as this
layer's roster).

## Dependency direction

This layer → `steam_reviews_framework` → `main_model`. Contrast towers
reuse the main model's tower skeleton and swap the loss
(`contrast_models/byol.py`, `arcface.py`); contrast heads reuse the
framework's two-phase head machinery and swap only the loss philosophy
(`contrast_heads/configs.py`). The framework never sees this layer —
deleting the whole of contrast_experiment does not affect champion
reproduction.

## w9/ — the full experiment suite

The complete campaign behind the paper's tables (fixed-split arms, 5-fold
grids, scaling sweep, anchor-supply/readout/temperature ablations) lives in
[`w9/`](w9/README.md). It is self-contained: workers, notebooks, and the
minimal model package they import, with no repository-sync or cloud-provider
operations.

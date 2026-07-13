# Pod route (RunPod, or any GPU cloud with a shared network volume)

Open **`w9_a100.ipynb`** (big jobs: anchor_cap >= 2048; also owns the
one-time full-pool asset build) or **`w9_l40.ipynb`** (small jobs: all g512
arms + all 5-fold CV) on a pod of the matching class and run it top to
bottom: dependency install → asset preparation (resumable; atomic rename +
READY marker) → local staging (`h5_staging.parallel_copy` thread-parallel
copy to local disk / shared memory) → the job queue (multi-machine-safe:
exclusive-create claim files, stale claims taken over after 12 h) →
completeness audit → `pod_selfstop` auto-shutdown (no idle burn).

The job table, labels and queue machinery live in **`w9_jobs.py`** — one
source of truth for both notebooks. Labels name the claim and result files,
so any mix of machines/notebooks (including runs from the retired
`w9_all.ipynb`) resume and skip each other's work. Each notebook has a
`SWEEP_*` flag to additionally drain the other class (single-pod mode).

- `w9_a100_worker.py` — fixed-split jobs (one arm = tower + per-checkpoint
  heads + vsel pick), rolling resume bundle: a restarted machine continues
  from its last checkpoint;
- `w9_cv_worker.py` — 5-fold CV jobs (fold-parameterized);
- both workers implement the SAME protocol (R60) as the local
  `steam_reviews_framework/train.py`; their json/npz artefact naming is
  compatible, so `contrast_experiment/report.py` aggregates pod and local
  results alike.

Multi-machine parallelism: open one copy of the notebook per machine; the
claim mechanism guarantees each job is picked up by exactly one machine.
The cheapest pattern is to let the first machine finish the "ONE-TIME
asset preparation" before starting the rest.

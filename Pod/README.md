# Pod Deployment — multi-VM coordinated sweep

`Pod/` is the **single template**. Run **`prepare_pods.ipynb` once, on the first VM** — it
installs git, pulls the repo, and generates the per-VM bundles `Pod_1 … Pod_7` at the
`/workspace` root (each with its `VM_NAME` = `VM1`…`VM7`). Because `/workspace` is shared,
one run makes **every VM's folder appear**; VM _N_ just opens `/workspace/Pod_N`. You only
maintain this one template — re-run the generator to push template changes to all bundles.

All VMs **share one `out_dir` on `/workspace`** and claim combos atomically, so no combo
is trained twice, the run is resumable, and an in-progress single-VM run **migrates
automatically** (a combo whose checkpoint already exists is recognised as done).

## Notebooks (run in this order)

- **`prepare_pods.ipynb`** (repo root, NOT in a bundle) — run once on the first VM to
  generate `/workspace/Pod_1..7` from this template. See the deploy sequence below.
- **`setup.ipynb`** — environment: gh CLI, clone/pull, `pip install`, flash-attn. Once per pod.
- **`prepare_training.ipynb`** — build/embed the H5 (**one VM only** — it writes the shared
  `embedding_h5.h5`; if it already exists, build/embed just skip) → **stage RESIDENT VECTORS**
  to local NVMe (`LOCAL_DATA`, default `/root/data`): a multiprocess read of the workspace H5
  into a raw fp16 `.dat` memmap (~150GiB) + a tiny companion mini-H5 (offsets + shape) →
  smoke-validate the pipeline (per-VM `cloud_smoke_<VM>`). Training then reads ONLY the local
  `.dat` + mini-H5 (one shared copy per VM via the OS page cache), never the big workspace H5.
  Gated on host RAM: if a VM is too small to keep the `.dat` resident, staging is skipped and
  training streams the workspace H5 instead.
- **`training.ipynb`** — the coordinated sweep. Registers this VM, claims combos, trains.
- **`check_paralle.ipynb`** — verify the *live* coordination (O_EXCL atomic on this mount,
  VM registry, migration OK). Run it **after** training has started.
- **`realtime_reader.ipynb`** — follow this VM's log + global coordinated progress.
- **`eval.ipynb`** — post-training: drain probes + final eval + archive (run on one VM).

## Deploy sequence

0. **First VM:** drag in + run **`prepare_pods.ipynb`** → generates `/workspace/Pod_1..7`
   (shared, so all VMs now have their folder).
1. **Every VM:** open its `/workspace/Pod_N` → `setup` → `prepare_training`.
2. **VM1 only:** start `training`.
3. **Run `check_paralle`** (on VM1, or any VM — it reads the shared dir): confirm
   **O_EXCL is atomic on MooseFS ✓**, **VM1 registered ✓**, **migration OK** (existing
   checkpoints recognized, none re-claimed) **✓**.
4. **If green → VM2…VM7:** start `training`.
5. **Re-run `check_paralle`** → see all 7 claiming, **no combo twice**.

## Multi-VM setup notes

- **`VM_NAME`** is set by `prepare_pods.ipynb` per bundle (`Pod_N` → `VM_N`) in
  `training.ipynb` / `prepare_training.ipynb` / `realtime_reader.ipynb` — don't hand-edit
  each folder; edit the `Pod/` template and re-run the generator. Duplicates auto-get a `_2`.
- **Shared `OUT_DIR`**: `VICReg_review/heads/cloud_full_sweep_a100` (all VMs). Each combo's
  files are written by exactly one VM → **outputs never overwrite each other**.
- **Per-VM log**: `/workspace/stable_query_latent_logs/pipeline_<VM>.log` — centrally
  readable, one file per VM, so logs never mix.
- **Machine-local scratch** (`calib.json`, job queue, ledger) lives under the system temp
  dir keyed by `VM_NAME` — never on the shared FS.
- **`LOCAL_DATA`** (default `/root/data`) is the machine-local big-file store for the resident
  vectors `.dat` + mini-H5 (`--local-data-dir`). Kept separate from `WORK_DIR` so the ~150GiB
  `.dat` sits on the large local volume while calib/queue/ledger stay small elsewhere. Only the
  `.dat` (+ tiny mini-H5) ends up local — no full local copy of the workspace H5.

## Download RunPod artifacts

Use the selective sync helper (`../tools/sync_runpod_artifacts.ps1`) instead of syncing the
whole bucket — it grabs only the expensive outputs (`text_h5.h5`, `embedding_h5.h5`, their
manifests, `VICReg_review/heads`, `stable_query_latent_artifacts`, logs):

```powershell
.\tools\sync_runpod_artifacts.ps1            # sync
.\tools\sync_runpod_artifacts.ps1 -DryRun    # preview
.\tools\sync_runpod_artifacts.ps1 -PrintOnly # print the aws s3 sync command only
```

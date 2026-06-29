# Pod Deployment Notes

This folder is a VM deployment bundle. Copy the files in this directory directly to
the root directory of the cloud virtual machine / pod before running the workflow.

## Files

- `setup.ipynb`: one-time or rerunnable environment setup.
- `run.ipynb`: main pipeline runner. Run this after `setup.ipynb`.
- `realtime_reader.ipynb`: log reader for checking progress while `run.ipynb` is
  running.
- `start_old.ipynb`: older startup notebook kept for reference.

## Workflow

1. Copy the contents of this `Pod/` directory to the VM root directory.
2. Open and run `setup.ipynb`.
3. Open and run `run.ipynb`.
4. When progress needs to be checked, open `realtime_reader.ipynb`.

`realtime_reader.ipynb` has two important cells:

- Cell 1 loads historical log output and manifest summaries.
- Cell 2 follows new log output in realtime. It starts after Cell 1's log offset,
  so it does not repeat the historical output already shown by Cell 1.

Interrupting Cell 2 only stops the log viewer. It does not stop the running
pipeline.

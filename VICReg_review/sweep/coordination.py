"""Decentralized combo coordination across VMs sharing ONE out_dir on the network FS.

PROTOTYPE -- not wired into the supervisor yet. Correctness is by construction:

* **claim = the status file itself.** Each combo has ONE status file
  ``<out_dir>/<combo_id>/status.json``, created with ``O_EXCL``. That exclusive create
  *is* the claim: two VMs racing for the same combo -> exactly one create succeeds, the
  other fails. This holds even with no liveness at all (the "static" guarantee). MooseFS
  serialises metadata through its master, so the exclusive create is atomic cross-client
  (unlike multi-writer append to one shared file). Status lives next to the checkpoint.
* **done = ``state == "done"`` in that status file, OR the checkpoint exists** (injected as
  ``done_fn``, so a combo finished by the OLD pipeline is recognised -> smooth migration).
* **reclaim** = a status file whose owner VM's lease has EXPIRED and that isn't done is
  freed via an atomic rename-aside (only one reclaimer can move it, so a live VM's fresh
  claim is never stolen).
* **fencing** = a worker re-checks ``owns()`` before committing, so a VM whose lease
  mis-fired (got reclaimed while stalled) can never write a duplicate result.

Liveness is a LEASE (flock was dropped -- unreliable cross-client on MooseFS): the VM writes
``expiry = now + lease`` into ``<out_dir>/VM_parallel/<vm>.json`` and a SEPARATE PROCESS
refreshes it (refresh << lease). A compute-bound training process can't starve a separate
refresher, so the lease lapses only when the VM is really gone.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

VM_DIR = "VM_parallel"
STATUS = "status.json"     # the exclusive CLAIM (write-once)
DONE = "done.json"         # the exclusive DONE marker (write-once)
FAILED = "failed.json"     # terminal FAILED marker (attempts exhausted here)


def _atomic_create(path, payload: dict) -> bool:
    """Exclusive create. True iff THIS caller created the file (the claim winner)."""
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except (FileExistsError, OSError):
        return False
    try:
        os.write(fd, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    finally:
        os.close(fd)
    return True


def _atomic_write(path, payload: dict) -> None:
    """Overwrite via tmp+rename so a reader never sees a torn file. Single-writer only.
    Retries on Windows' transient "target open by a reader" error (POSIX renames fine)."""
    path = Path(path)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}.{int(time.time() * 1e6)}"
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    for _ in range(100):
        try:
            os.replace(str(tmp), str(path))
            return
        except PermissionError:
            time.sleep(0.003)
    os.replace(str(tmp), str(path))


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------- liveness
def _lease_refresh_loop(vm_file, lease, refresh, parent_pid):
    """Separate-process refresher: bump the lease expiry every ``refresh`` seconds.
    Dies with the parent (PDEATHSIG on Linux + a parent-PID poll backstop), so a
    dead VM's lease actually lapses and its claims become reclaimable."""
    try:
        import ctypes, signal
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass
    vm_file = Path(vm_file)
    while True:
        if parent_pid and not _pid_alive(parent_pid):
            return
        rec = _read_json(vm_file) or {}
        rec["expiry"] = time.time() + lease
        try:
            _atomic_write(vm_file, rec)
        except OSError:
            pass
        time.sleep(refresh)


class LeaseLiveness:
    """Explicit-expiry lease refreshed by a SEPARATE PROCESS (refresh << lease)."""

    def __init__(self, lease: float = 600.0, refresh: float = 120.0, use_process: bool = True):
        self.lease = float(lease)
        self.refresh = float(refresh)
        self.use_process = bool(use_process)
        self._proc = None
        self._file = None

    def start(self, vm_file) -> None:
        self._file = Path(vm_file)
        self.refresh_now()
        if self.use_process:
            import multiprocessing as mp
            self._proc = mp.Process(target=_lease_refresh_loop,
                                    args=(str(self._file), self.lease, self.refresh, os.getpid()),
                                    daemon=True)
            self._proc.start()

    def refresh_now(self) -> None:
        if self._file is not None:
            rec = _read_json(self._file) or {}
            rec["expiry"] = time.time() + self.lease
            _atomic_write(self._file, rec)

    def stop(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None

    def revoke(self) -> None:
        """Expire the lease NOW (clean shutdown). Without this a killed
        supervisor's claims stay 'live' for the remaining lease window
        (<=10 min): a same-name restart gets a _2 suffix, and peers slide to
        lower-priority combos while apparently-live claims block the queue --
        each mis-taken combo costs its full training time. Revoking makes the
        claims reclaimable immediately; the lease window remains only for
        hard deaths (SIGKILL, OOM-killer, pulled pods)."""
        self.stop()
        if self._file is None:
            return
        rec = _read_json(self._file) or {}
        rec["expiry"] = 0.0
        rec["revoked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            _atomic_write(self._file, rec)
        except OSError:
            pass

    def alive(self, vm_file) -> bool:
        rec = _read_json(vm_file)
        return bool(rec) and time.time() < float(rec.get("expiry", 0))


class HeartbeatLiveness:
    """mtime-freshness lease bumped by a background thread. Lightweight; used in tests."""

    def __init__(self, interval: float = 5.0, stale_after: float = 60.0):
        self.interval = float(interval)
        self.stale_after = float(stale_after)
        self._stop = None
        self._file = None

    def start(self, vm_file) -> None:
        import threading
        self._file = Path(vm_file)
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                os.utime(self._file, None)
            except OSError:
                pass

    def refresh_now(self) -> None:
        if self._file is not None:
            try:
                os.utime(self._file, None)
            except OSError:
                pass

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()

    def revoke(self) -> None:
        """mtime-based equivalent of LeaseLiveness.revoke: age the file out."""
        self.stop()
        if self._file is not None:
            try:
                os.utime(self._file, (0.0, 0.0))
            except OSError:
                pass

    def alive(self, vm_file) -> bool:
        try:
            return (time.time() - Path(vm_file).stat().st_mtime) <= self.stale_after
        except OSError:
            return False


# --------------------------------------------------------------------------- coordinator
class Coordinator:
    """One VM's view of the shared sweep. ``root`` is the shared out_dir (heads).
    Per-combo state is ``root/<combo_id>/status.json`` (the exclusive claim); VM leases
    are ``root/VM_parallel/<vm>.json``."""

    def __init__(self, root, vm_name: str, *, done_fn=None, liveness=None, info=None):
        self.root = Path(root)
        self.vms = self.root / VM_DIR
        self.vms.mkdir(parents=True, exist_ok=True)
        self._done_fn = done_fn or (lambda cid: False)
        self.liveness = liveness or LeaseLiveness()
        self.vm_name = self._register(vm_name, info or {})

    # --- VM registry (collision-safe) -------------------------------------
    def _vm_file(self, vm) -> Path:
        return self.vms / f"{vm}.json"

    def _vm_alive(self, vm) -> bool:
        return self.liveness.alive(self._vm_file(vm))

    def _vm_record(self, vm) -> dict:
        return _read_json(self._vm_file(vm)) or {}

    def _claim_is_live(self, rec) -> bool:
        """A claim is protected only by ITS OWNER INCARNATION's lease.

        The vm NAME holding a fresh lease is not enough: after a same-name
        restart (old lease lapsed, _register reused the name) the new
        incarnation's lease must not resurrect the dead predecessor's claims.
        The claim records the writer's pid and the registry the current
        incarnation's pid -- a mismatch means the claim belongs to a previous
        life and is reclaimable. Legacy records without pids fall back to the
        lease-only rule."""
        vm = rec.get("vm") if rec else None
        if not vm or not self._vm_alive(vm):
            return False
        claim_pid = rec.get("pid")
        reg_pid = self._vm_record(vm).get("pid")
        if claim_pid is None or reg_pid is None:
            return True
        try:
            return int(claim_pid) == int(reg_pid)
        except (TypeError, ValueError):
            return True

    def alive_vms(self) -> list:
        """(vm_name, info) for every registered VM whose lease is still fresh. Used
        for capability routing: each VM sees everyone's GPU/VRAM to decide who runs
        the monster combos."""
        out = []
        for f in self.vms.glob("*.json"):
            if self.liveness.alive(f):
                rec = _read_json(f) or {}
                out.append((rec.get("vm") or f.stem, rec.get("info", {})))
        return out

    def _register(self, base: str, info: dict) -> str:
        name, i = base, 1
        while self._vm_file(name).exists() and self._vm_alive(name):
            i += 1
            name = f"{base}_{i}"
        _atomic_write(self._vm_file(name),
                      {"vm": name, "pid": os.getpid(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "info": info})
        self.liveness.start(self._vm_file(name))
        return name

    def close(self) -> None:
        """Clean shutdown: REVOKE the lease (not just stop refreshing it) so our
        claims become reclaimable immediately -- no <=10 min zombie window, no
        _2 suffix on restart, no peers wandering off to lower-priority combos."""
        revoke = getattr(self.liveness, "revoke", None)
        if revoke is not None:
            revoke()
        else:
            self.liveness.stop()

    # --- per-combo status (claim + done, both write-once) -----------------
    def _status_file(self, cid) -> Path:
        return self.root / str(cid) / STATUS

    def _done_file(self, cid) -> Path:
        return self.root / str(cid) / DONE

    def _failed_file(self, cid) -> Path:
        return self.root / str(cid) / FAILED

    def _status(self, cid):
        return _read_json(self._status_file(cid))

    def claim_owner(self, cid):
        rec = self._status(cid)
        return rec.get("vm") if rec else None

    def claim_lane(self, cid):
        rec = self._status(cid)
        return rec.get("lane") if rec else None

    def owns(self, cid, lane=None) -> bool:
        """Fencing check: does THIS (vm, lane) still hold this claim? The lane
        sub-identity keeps two GPU lanes of one VM from resuming each other's
        in-flight claim (they share vm_name); liveness/reclaim stays per-VM."""
        rec = self._status(cid)
        return bool(rec) and rec.get("vm") == self.vm_name and rec.get("lane") == lane

    def is_done(self, cid) -> bool:
        # checkpoint (authoritative; recognises the OLD pipeline -> smooth migration)
        # OR the exclusive done marker.
        return bool(self._done_fn(cid)) or self._done_file(cid).exists()

    def is_terminal(self, cid) -> bool:
        """Done OR failed here -- either way this VM won't re-claim it, and a drain
        loop can treat it as finished. (A more-capable VM could clear failed.json and
        retry; not done yet.)"""
        return self.is_done(cid) or self._failed_file(cid).exists()

    def mark_failed(self, cid, error="") -> None:
        _atomic_create(self._failed_file(cid), {"vm": self.vm_name, "error": str(error)[:500],
                                                "ts": time.time()})

    def try_claim(self, cid, lane=None) -> bool:
        sf = self._status_file(cid)
        sf.parent.mkdir(parents=True, exist_ok=True)
        return _atomic_create(sf, {"vm": self.vm_name, "lane": lane,
                                   "pid": os.getpid(), "ts": time.time()})

    def mark_done(self, cid) -> None:
        _atomic_create(self._done_file(cid), {"vm": self.vm_name, "done_ts": time.time()})

    def fenced_done(self, cid, lane=None) -> bool:
        """Commit ONLY if we still own the claim (a lost claim -> result discarded)."""
        if not self.owns(cid, lane):
            return False
        self.mark_done(cid)
        return True

    def release(self, cid) -> None:
        """Give up our OWN, not-done claim (on failure/exit) so a peer can take it."""
        if self.owns(cid) and not self.is_done(cid):
            try:
                self._status_file(cid).unlink()
            except OSError:
                pass

    def _refresh_claim(self, cid, lane) -> bool:
        """Stamp OUR incarnation onto a claim we are resuming: a claim inherited
        from a same-name predecessor still carries the old pid, so without the
        refresh every peer would see it as dead and could reclaim it while we
        train.

        Re-verify ownership IMMEDIATELY before the overwrite: at tail-restart a
        peer may reclaim the dead-generation claim in the same instant, and a
        blind rewrite would clobber its fresh claim -- both sides then train
        the combo (observed: duelling checkpoint writes every flush pulse).
        The remaining ms-window is settled by the fenced_done commit."""
        rec = self._status(cid)
        if not rec or rec.get("vm") != self.vm_name or rec.get("lane") != lane:
            return False                       # lost to a peer in the same instant
        _atomic_write(self._status_file(cid),
                      {"vm": self.vm_name, "lane": lane, "pid": os.getpid(), "ts": time.time()})
        return True

    def reconcile_own_claims(self, grid_ids, lanes) -> int:
        """Startup pass over claims left by a SAME-NAME predecessor (pid differs).

        Keep the ones our current lanes will naturally resume (combo still in
        OUR grid and its lane still exists here); RELEASE the rest -- combos
        masked out of the grid or lanes this incarnation no longer runs would
        otherwise look in-flight forever (nobody walks them, and peers used to
        see the fresh same-name lease and never reclaim)."""
        released = 0
        me = os.getpid()
        for cdir in self.root.iterdir():
            if cdir.name == VM_DIR or not cdir.is_dir():
                continue
            rec = _read_json(self._status_file(cdir.name)) or {}
            if rec.get("vm") != self.vm_name:
                continue
            try:
                if int(rec.get("pid")) == me:
                    continue
            except (TypeError, ValueError):
                pass
            if self.is_done(cdir.name):
                continue
            if cdir.name in grid_ids and rec.get("lane") in lanes:
                continue                     # our lane resumes + refreshes it
            try:
                self._status_file(cdir.name).unlink()
                released += 1
            except OSError:
                pass
        return released

    def _reclaim_dead(self, cid) -> bool:
        """Free a status file whose owner incarnation is gone, via atomic rename-aside."""
        sf = self._status_file(cid)
        rec = self._status(cid) or {}
        owner = rec.get("vm")
        if not owner or owner == self.vm_name or self._claim_is_live(rec) or self.is_done(cid):
            return False
        aside = sf.parent / f".{STATUS}.stale.{self.vm_name}.{int(time.time() * 1000)}"
        try:
            os.rename(str(sf), str(aside))       # atomic CAS: exactly one reclaimer wins
        except OSError:
            return False
        try:
            aside.unlink()
        except OSError:
            pass
        return True

    def reclaim_stale(self) -> int:
        freed = 0
        for cdir in self.root.iterdir():
            if cdir.name == VM_DIR or not cdir.is_dir():
                continue
            if self._reclaim_dead(cdir.name):
                freed += 1
        return freed

    def next_claim(self, ordered_combo_ids, fits_fn=None, lane=None):
        """Claim the next combo THIS (vm, lane) should run, in the given (capability)
        order. Resumes only OUR lane's own in-flight claim; a claim held by any live
        VM (another lane or another VM) is left alone; a dead VM's claim is reclaimed."""
        for cid in ordered_combo_ids:
            if self.is_terminal(cid):
                continue
            rec = self._status(cid)
            owner = rec.get("vm") if rec else None
            if owner is not None:
                if owner == self.vm_name and rec.get("lane") == lane:
                    # Resume OUR lane's claim (possibly inherited from a
                    # same-name predecessor) and stamp it with THIS pid so
                    # peers stop seeing it as a dead incarnation's claim. If a
                    # peer snatched it in the same instant, walk on.
                    if self._refresh_claim(cid, lane):
                        return cid
                    continue
                if self._claim_is_live(rec):
                    continue                      # a live incarnation has it
                self._reclaim_dead(cid)           # dead incarnation -> free the slot
            if fits_fn is not None and not fits_fn(cid):
                continue                          # too big here -> leave for a bigger VM
            if self.try_claim(cid, lane):
                return cid
        return None

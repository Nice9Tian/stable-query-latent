"""Resident-vectors staging + memmap gather correctness.

The .dat must be a bit-faithful copy of the H5 'vectors', and reading views via the
memmap must produce IDENTICAL tensors to reading them from the H5 (resident mode is a
pure I/O change -- same trained artifact)."""

import os
import tempfile

import h5py
import numpy as np
import torch

from VICReg_review import train_vicreg_review_h5 as trainer


def _close_memmaps(*arrays):
    """Release memmap file handles so Windows can delete the temp .dat."""
    for arr in arrays:
        mm = getattr(arr, "_mmap", None)
        if mm is not None:
            mm.close()
    for arr in trainer._PROC_MEMMAP.values():
        mm = getattr(arr, "_mmap", None)
        if mm is not None:
            mm.close()
    trainer._PROC_MEMMAP.clear()


def _tmpdir():
    # ignore_cleanup_errors: memmap teardown on Windows can lag the rmtree.
    return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


def _build_h5(path):
    """Tiny H5 with the datasets the trainer read-path needs. Chunked, like prod."""
    dim = 8
    rng = np.random.default_rng(0)
    per_game_reviews = [[3, 2], [1, 4], [2]]          # 3 games; review sentence counts
    review_offsets = [0]
    game_review_offsets = [0]
    blocks = []
    review_count = 0
    for reviews in per_game_reviews:
        for sentences in reviews:
            blocks.append(rng.standard_normal((sentences, dim)).astype(np.float16))
            review_offsets.append(review_offsets[-1] + sentences)
            review_count += 1
        game_review_offsets.append(review_count)
    vectors = np.concatenate(blocks, axis=0)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("vectors", data=vectors, chunks=(4, dim))
        h5.create_dataset("review_offsets", data=np.asarray(review_offsets, np.int64))
        h5.create_dataset("game_review_offsets", data=np.asarray(game_review_offsets, np.int64))
        h5.create_dataset("game_names",
                          data=np.asarray(["g0", "g1", "g2"], dtype=h5py.string_dtype("utf-8")))
        h5.attrs["input_dim"] = dim
    return vectors


def test_stage_resident_vectors_roundtrips_and_descriptor():
    with _tmpdir() as d:
        h5p = os.path.join(d, "emb.h5")
        vectors = _build_h5(h5p)
        # chunk_rows=4 over 12 rows -> 3 disjoint ranges (multi-range write path);
        # workers=1 keeps it in-process (a spawn pool would re-exec the test runner).
        desc = trainer.stage_resident_vectors(h5p, work_dir=d, chunk_rows=4, workers=1)
        path, (rows, dim), dtype_name = desc
        assert (rows, dim) == vectors.shape
        assert dtype_name == "float16"
        mm = np.memmap(path, dtype=np.float16, mode="r", shape=(rows, dim))
        assert np.array_equal(np.asarray(mm), vectors)             # bit-faithful dump
        assert trainer.resident_descriptor(path, h5p) == desc      # sidecar round-trips
        assert trainer.stage_resident_vectors(h5p, work_dir=d) == desc   # idempotent
        _close_memmaps(mm)


def test_memmap_views_match_h5_views_bit_for_bit():
    with _tmpdir() as d:
        h5p = os.path.join(d, "emb.h5")
        _build_h5(h5p)
        desc = trainer.stage_resident_vectors(h5p, work_dir=d, workers=1)
        f16 = np.dtype("float16")
        try:
            with h5py.File(h5p, "r") as h5:
                trainer._VECTORS_DAT = None                        # read from H5
                a1, b1 = trainer.load_game_views(h5, 1, 0.6, np.random.default_rng(42), f16, False)
                trainer._VECTORS_DAT = desc                        # read from memmap
                a2, b2 = trainer.load_game_views(h5, 1, 0.6, np.random.default_rng(42), f16, False)
        finally:
            trainer._VECTORS_DAT = None
            _close_memmaps()
        assert torch.equal(a1, a2)
        assert torch.equal(b1, b2)


def test_mini_h5_validates_is_tiny_and_drives_reads_without_big_h5():
    with _tmpdir() as d:
        h5p = os.path.join(d, "emb.h5")
        _build_h5(h5p)
        desc = trainer.stage_resident_vectors(h5p, work_dir=d, workers=1)
        meta_h5 = os.path.join(d, "emb.meta.h5")
        trainer.write_offsets_h5(h5p, meta_h5)
        with h5py.File(meta_h5, "r") as m:
            trainer.validate_training_h5(m, meta_h5)                # must not raise
            assert tuple(m["vectors"].shape) == (12, 8)            # shape preserved for .shape reads
            assert m["vectors"].id.get_storage_size() == 0         # ...but no bulk data stored
            assert list(m["game_review_offsets"][:]) == [0, 2, 4, 5]
        assert os.path.getsize(meta_h5) < os.path.getsize(h5p)     # a fraction of the source
        # reads via (mini-H5 offsets + .dat vectors) == reads via the full source H5
        f16 = np.dtype("float16")
        try:
            with h5py.File(h5p, "r") as full:
                trainer._VECTORS_DAT = None
                a1, b1 = trainer.load_game_views(full, 1, 0.6, np.random.default_rng(7), f16, False)
            with h5py.File(meta_h5, "r") as mini:
                trainer._VECTORS_DAT = desc
                a2, b2 = trainer.load_game_views(mini, 1, 0.6, np.random.default_rng(7), f16, False)
        finally:
            trainer._VECTORS_DAT = None
            _close_memmaps()
        assert torch.equal(a1, a2)
        assert torch.equal(b1, b2)

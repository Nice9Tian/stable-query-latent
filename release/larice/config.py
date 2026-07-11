# -*- coding: utf-8 -*-
"""LariceConfig — one dataclass builds one larice tower
(larice = Latent Represent I-CE).

The model package is task-agnostic: it knows nothing about Steam, corpora,
or evaluation protocols. Everything a downstream task can tune lives here.
"""
from dataclasses import dataclass


@dataclass
class LariceConfig:
    # ---- architecture ----
    num_queries: int = 4       # N latent query slots
    dim_model: int = 128       # DM: query/attention/output width per slot
    num_heads: int = 4         # attention heads
    input_dim: int = 1024      # D_in: dimensionality of upstream embeddings
    hidden: int = 256          # readout MLP hidden width
    readout: str = "concat"    # "concat" (default, general representation)
    #                          # | "pool" (mean over slots; better for
    #                          #   name-recall-style retrieval, see README)

    # ---- champion loss (I-CE with CE gating) ----
    num_views: int = 4         # NV: views per data item fed to the loss
    tau_mode: str = "frozen"   # "frozen" | "learnable"
    tau: float = 0.02          # CE temperature (init value when learnable)
    inv_weight: float = 2.0    # I: invariance weight across the view axis
    #                          # CE gate: a per-item bool mask passed at loss
    #                          # call time — CE fires only where gate is True,
    #                          # I always fires (champion recipe).

    @property
    def out_dim(self) -> int:
        return (self.num_queries * self.dim_model if self.readout == "concat"
                else self.dim_model)

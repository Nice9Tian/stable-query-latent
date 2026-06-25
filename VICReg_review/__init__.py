"""VICReg training components for game-review latent arrays."""

from .model import (
    GameCentroidExpander,
    GradientReversal,
    HierarchicalLatentArrayMLP,
    LatentArrayMLP,
    Latent_Array_MLP,
    Mlp4SentimentHead,
    SentimentAdversarialLoss,
    game_centroid,
    load_mlp4_a_sentiment_head,
    vicreg_centroid_loss,
    vicreg_loss,
)

__all__ = [
    "GameCentroidExpander",
    "GradientReversal",
    "HierarchicalLatentArrayMLP",
    "LatentArrayMLP",
    "Latent_Array_MLP",
    "Mlp4SentimentHead",
    "SentimentAdversarialLoss",
    "game_centroid",
    "load_mlp4_a_sentiment_head",
    "vicreg_centroid_loss",
    "vicreg_loss",
]

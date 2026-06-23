"""VICReg training components for game-review latent arrays."""

from .model import (
    GradientReversal,
    LatentArrayMLP,
    Latent_Array_MLP,
    Mlp4SentimentHead,
    SentimentAdversarialLoss,
    load_mlp4_a_sentiment_head,
    vicreg_loss,
)

__all__ = [
    "GradientReversal",
    "LatentArrayMLP",
    "Latent_Array_MLP",
    "Mlp4SentimentHead",
    "SentimentAdversarialLoss",
    "load_mlp4_a_sentiment_head",
    "vicreg_loss",
]

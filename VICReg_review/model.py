"""Latent-array VICReg model with a GRL sentiment adversary.

The encoder projects the 1024-d input down to a wide latent_dim (256 by default),
runs cross- and self-attention at that width, then funnels each latent vector
down to a small output_dim (256 -> 128 -> 64 -> 32 -> 18 by default). VICReg runs
on these compact codes. Because the frozen SST MLP4-A head still expects 1024-d
inputs, the adversary holds a learnable up-projection probe (output_dim -> 1024)
placed AFTER the GRL, so the encoder is always the adversarial party.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class _GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def gradient_reverse(x, lambda_=1.0):
    return _GradientReverseFn.apply(x, float(lambda_))


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x):
        return gradient_reverse(x, self.lambda_)


def _make_funnel_mlp(dims, dropout):
    """Sequential Linear funnel through dims, e.g. [256, 128, 64, 32, 18].

    GELU + Dropout between hidden layers; the final projection is raw (no
    activation, no norm) so VICReg's variance term controls the output scale.
    """
    layers = []
    for index in range(len(dims) - 1):
        layers.append(nn.Linear(dims[index], dims[index + 1]))
        if index < len(dims) - 2:
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class LatentArrayMLP(nn.Module):
    """Minimal shared-weight view encoder for review subsets.

    One cross-attention layer (learnable query array attends to the sentences),
    no residuals and no extra blocks, then a per-latent funnel down to output_dim.

    Input:
        x: (batch, sentence_count, input_dim)
        key_padding_mask: optional bool mask, True where x is padding

    Output:
        (batch, num_latents, output_dim), default (batch, 256, 18)
    """

    def __init__(
        self,
        input_dim=1024,
        latent_dim=256,
        num_latents=256,
        num_heads=8,
        dropout=0.1,
        output_dim=18,
        reduce_hidden=(128, 64, 32),
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.num_latents = int(num_latents)
        self.output_dim = int(output_dim)

        self.input_norm = nn.LayerNorm(input_dim)
        if input_dim == latent_dim:
            self.input_proj = nn.Identity()
        else:
            self.input_proj = nn.Linear(input_dim, latent_dim)

        self.latent_array = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)
        self.query_norm = nn.LayerNorm(latent_dim)
        self.context_norm = nn.LayerNorm(latent_dim)
        self.cross_attention = nn.MultiheadAttention(
            latent_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(latent_dim)
        self.reduce = _make_funnel_mlp([latent_dim, *reduce_hidden, self.output_dim], dropout)

    def forward(self, x, key_padding_mask=None):
        context = self.context_norm(self.input_proj(self.input_norm(x)))
        queries = self.latent_array.unsqueeze(0).expand(x.size(0), -1, -1)

        latents, _ = self.cross_attention(
            query=self.query_norm(queries),
            key=context,
            value=context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        latents = self.output_norm(latents)
        return self.reduce(latents)


Latent_Array_MLP = LatentArrayMLP


class TagRegressionHead(nn.Module):
    """Validation-only probe: map a frozen encoder code to tag signals.

    Input is the encoder output (batch, num_latents, latent_out_dim). By default
    the latent set is flattened (num_latents * latent_out_dim) so the head sees
    the full representation; use pool="mean" to average over latents instead.
    pool="stats" concatenates mean/std/max/min over latent slots, which gives a
    compact probe for small validation sets.

    Outputs a dict with raw presence logits and raw-count regression logits.
    Apply sigmoid() to presence_logits for tag-existence probabilities. Apply
    softplus() and expm1() to count_logits to recover non-negative tag counts.
    This head is never part of the VICReg loss path -- it is trained separately
    on a frozen encoder.
    """

    def __init__(
        self,
        num_tags,
        num_latents=256,
        latent_out_dim=18,
        hidden_dims=(256, 128),
        dropout=0.1,
        pool="flatten",
    ):
        super().__init__()
        if pool not in ("flatten", "mean", "stats"):
            raise ValueError("pool must be 'flatten', 'mean', or 'stats'.")
        self.pool = pool
        self.num_tags = int(num_tags)
        if pool == "flatten":
            in_dim = num_latents * latent_out_dim
        elif pool == "stats":
            in_dim = latent_out_dim * 4
        else:
            in_dim = latent_out_dim
        layers = []
        prev = in_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.LayerNorm(prev),
                nn.Linear(prev, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = hidden_dim
        layers += [nn.LayerNorm(prev)]
        self.trunk = nn.Sequential(*layers)
        self.presence = nn.Linear(prev, self.num_tags)
        self.count = nn.Linear(prev, self.num_tags)

    def forward(self, feats):
        if self.pool == "flatten":
            x = feats.flatten(start_dim=1)
        elif self.pool == "stats":
            x = torch.cat(
                [
                    feats.mean(dim=1),
                    feats.std(dim=1, unbiased=False),
                    feats.amax(dim=1),
                    feats.amin(dim=1),
                ],
                dim=1,
            )
        else:
            x = feats.mean(dim=1)
        x = self.trunk(x)
        return {
            "presence_logits": self.presence(x),
            "count_logits": self.count(x),
        }


class Mlp4SentimentHead(nn.Module):
    """SST MLP4-A: 1024 -> 128 -> 32 -> 8 -> 1 with sigmoid output."""

    def __init__(self, input_dim=1024, hidden_dims=(128, 32, 8), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(prev, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            prev = hidden_dim
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_mlp4_a_sentiment_head(checkpoint_path, map_location="cpu", freeze=True):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model = Mlp4SentimentHead()
    model.load_state_dict(state_dict)
    model.eval()
    if freeze:
        for param in model.parameters():
            param.requires_grad_(False)
    return model


def _off_diagonal(x):
    rows, cols = x.shape
    if rows != cols:
        raise ValueError("_off_diagonal expects a square matrix.")
    return x.flatten()[:-1].view(rows - 1, cols + 1)[:, 1:].flatten()


def vicreg_loss(
    z_a,
    z_b,
    invariance_weight=25.0,
    variance_weight=25.0,
    covariance_weight=1.0,
    eps=1e-4,
):
    """VICReg loss for two latent-array views.

    Invariance is computed on matching latent positions. Variance and covariance
    treat all latent vectors in the batch as the sample axis, so the covariance
    matrix is output_dim x output_dim (18 x 18 by default), not the flattened
    num_latents x output_dim feature set.
    """

    if z_a.shape != z_b.shape:
        raise ValueError(f"VICReg views must have matching shapes, got {z_a.shape} and {z_b.shape}.")

    z_a = z_a.float()
    z_b = z_b.float()
    repr_loss = F.mse_loss(z_a, z_b)
    flat_a = z_a.reshape(-1, z_a.size(-1))
    flat_b = z_b.reshape(-1, z_b.size(-1))

    def variance_term(z):
        std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
        return torch.mean(F.relu(1.0 - std))

    def covariance_term(z):
        sample_count = z.size(0)
        if sample_count < 2:
            return z.new_tensor(0.0)
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (sample_count - 1)
        return _off_diagonal(cov).pow(2).sum() / z.size(1)

    std_loss = 0.5 * (variance_term(flat_a) + variance_term(flat_b))
    cov_loss = 0.5 * (covariance_term(flat_a) + covariance_term(flat_b))
    total = (
        invariance_weight * repr_loss
        + variance_weight * std_loss
        + covariance_weight * cov_loss
    )
    return {
        "loss": total,
        "invariance": repr_loss,
        "variance": std_loss,
        "covariance": cov_loss,
    }


class SentimentAdversarialLoss(nn.Module):
    """GRL loss that pushes latent vectors toward SST-head uncertainty.

    The SST head is a frozen 0..1 regressor. We use Bernoulli entropy as a
    confidence surrogate. Minimizing entropy after a GRL makes the encoder ascend
    that entropy, so the frozen sentiment head is driven toward uncertainty.
    """

    def __init__(
        self,
        sentiment_head,
        input_dim=18,
        probe_hidden=256,
        probe_dim=1024,
        grl_lambda=1.0,
        eps=1e-6,
        normalize=True,
    ):
        super().__init__()
        self.sentiment_head = sentiment_head
        self.grl = GradientReversal(grl_lambda)
        # Learnable up-projection probe, placed AFTER the GRL: it tries to recover
        # sentiment confidence from the compact encoder code, while the GRL makes
        # the encoder fight it. Its own gradients are NOT reversed. Biases are
        # disabled so the probe cannot use a learned constant channel shortcut.
        self.probe = nn.Sequential(
            nn.Linear(input_dim, probe_hidden, bias=False),
            nn.GELU(),
            nn.Linear(probe_hidden, probe_dim, bias=False),
        )
        self.eps = eps
        self.normalize = normalize

    def forward(self, latents):
        # Run the probe + frozen head + entropy in fp32: normalize() and the
        # Bernoulli entropy overflow to NaN easily under AMP fp16.
        with torch.amp.autocast("cuda", enabled=False):
            flat = latents.reshape(-1, latents.size(-1)).float()
            up = self.probe(self.grl(flat))
            if self.normalize:
                up = F.normalize(up, p=2, dim=-1)
            pred = self.sentiment_head(up).float().clamp(self.eps, 1.0 - self.eps)
            entropy = -(pred * pred.log() + (1.0 - pred) * (1.0 - pred).log())
            loss = entropy.mean()
            with torch.no_grad():
                stats = {
                    "sentiment_mean": pred.mean(),
                    "sentiment_std": pred.std(unbiased=False),
                    "sentiment_entropy": entropy.mean(),
                }
        return loss, stats

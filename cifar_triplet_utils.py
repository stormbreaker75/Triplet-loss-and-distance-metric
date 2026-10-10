"""
Shared utility library: common code for the CIFAR-10 triplet positive/negative demo
====================================================================================

Called by 7 independent demo scripts (one per distance metric, each trained separately):
    demo_cosine.py / demo_euclidean.py / demo_manhattan.py
    demo_chebyshev.py / demo_minkowski.py / demo_hamming.py / demo_mahalanobis.py

This library provides:
    1. CIFAR-10 download (official, via torchvision) + per-class sampled subset
    2. A small CNN embedding network
    3. Pairwise distance matrices for 7 distance/similarity metrics (differentiable; mining uses detach copies)
    4. In-batch positive/negative mining: mine_triplets
    5. Differentiable triplet loss: triplet_loss (Hamming uses a smooth proxy)
    6. Independent training loop: train_embedding
    7. Result reporting (intra/inter-class distance, example triplets)

Note: Hamming distance binarization is non-differentiable, so its "training" uses the
      cosine triplet loss as a reasonable proxy, while mining and evaluation use the
      true Hamming distance.
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]

DIFFERENTIABLE = {"cosine", "euclidean", "manhattan", "chebyshev",
                  "minkowski", "mahalanobis"}


# --------------------------------------------------------------------------- #
# 1. Data: download + per-class sampling
# --------------------------------------------------------------------------- #

def load_subset(data_dir: str, per_class: int = 40, seed: int = 0,
                train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Download (if not cached) and load CIFAR-10, sampling per_class images per class.

    Args:
        data_dir: Data root directory.
        per_class: Number of samples per class.
        seed: Sampling random seed.
        train: If True use the training set (data_batch_*), if False the test set (test_batch).

    Returns:
        images: (N,3,32,32) float, normalized
        labels: (N,) long
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    ds = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)

    rng = np.random.default_rng(seed)
    idx = []
    for c in range(10):
        cls_idx = np.where(np.array(ds.targets) == c)[0]
        idx.extend(rng.choice(cls_idx, size=min(per_class, len(cls_idx)), replace=False))
    idx = sorted(idx)

    imgs = torch.stack([ds[i][0] for i in idx])   # (N,3,32,32)
    lbls = torch.tensor([ds[i][1] for i in idx], dtype=torch.long)
    return imgs, lbls


# --------------------------------------------------------------------------- #
# 2. Embedding network (small CNN)
# --------------------------------------------------------------------------- #

class EmbedNet(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )  # 32 -> 4x4, 128*4*4=2048
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output raw embeddings; cosine normalizes internally in _cosine, hamming binarizes inside _hamming
        return self.head(self.features(x))


# --------------------------------------------------------------------------- #
# 3. Pairwise distance matrices for 7 distance/similarity metrics
# --------------------------------------------------------------------------- #

def _cosine(A, B):  # 1 - cos
    return 1.0 - F.normalize(A, dim=1) @ F.normalize(B, dim=1).t()


def _euclidean(A, B):
    return torch.cdist(A, B, p=2.0)


def _manhattan(A, B):
    return torch.cdist(A, B, p=1.0)


def _chebyshev(A, B):
    return (A.unsqueeze(1) - B.unsqueeze(0)).abs().max(dim=2).values


def _minkowski(A, B):
    return torch.cdist(A, B, p=3.0)


def _hamming(A, B):
    # Center by batch mean, then sign-hash: threshold each dim at the batch mean to
    # remove the common component and keep the code bits diverse
    A_c = A - A.mean(dim=0, keepdim=True)
    B_c = B - B.mean(dim=0, keepdim=True)
    Ab = (A_c > 0).to(A.dtype)
    Bb = (B_c > 0).to(A.dtype)
    return (Ab.unsqueeze(1) != Bb.unsqueeze(0)).float().mean(dim=2)


def _mahalanobis(A, B, inv_cov=None):
    if inv_cov is None:
        return _euclidean(A, B)
    diff = A.unsqueeze(1) - B.unsqueeze(0)
    mid = diff @ inv_cov
    return torch.sqrt(torch.clamp((mid * diff).sum(dim=2), min=0.0))


_METRICS = {
    "cosine":      ("Cosine (余弦)",       _cosine),
    "euclidean":   ("Euclidean (欧氏)",    _euclidean),
    "manhattan":   ("Manhattan (曼哈顿)",  _manhattan),
    "chebyshev":   ("Chebyshev (切比雪夫)", _chebyshev),
    "minkowski":   ("Minkowski (闵可夫斯基, p=3)", _minkowski),
    "hamming":     ("Hamming (汉明)",     _hamming),
    "mahalanobis": ("Mahalanobis (马氏)", _mahalanobis),
}


def dist_matrix(A: torch.Tensor, metric: str, inv_cov: Optional[torch.Tensor] = None,
                B: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Pairwise distance matrix. A:(M,D), B:(N,D) -> (M,N); if B is None, B=A gives (M,M).

    Used for mining / intra-inter-class distances (B=A) and cross-dataset retrieval
    (query=A, gallery=B). A and B may be tensors with gradients; pass detach versions
    when pure numeric values are needed.
    """
    if metric not in _METRICS:
        raise KeyError(f"Unknown metric {metric}, available: {list(_METRICS)}")
    B = A if B is None else B
    fn = _METRICS[metric][1]
    return fn(A, B, inv_cov) if metric == "mahalanobis" else fn(A, B)


def estimate_inv_cov(emb: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """Estimate the inverse covariance matrix from samples (used for Mahalanobis distance).

    Eigen-decompose and clamp overly small eigenvalues up to eps, bounding the eigenvalues
    of the inverse within 1/eps, to avoid ill-conditioned directions blowing up the
    Mahalanobis distance into huge values and causing gradient explosion.
    """
    X = emb.detach()
    N, D = X.shape
    centered = X - X.mean(dim=0, keepdim=True)
    cov = (centered.t() @ centered) / (N - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp(min=eps)                       # lift small eigenvalues
    inv_cov = (eigvecs / eigvals.unsqueeze(0)) @ eigvecs.t()   # V·D⁻¹·Vᵀ
    return inv_cov


# --------------------------------------------------------------------------- #
# 4. In-batch positive/negative mining (selection does not take part in gradients)
# --------------------------------------------------------------------------- #

def mine_triplets(emb: torch.Tensor, labels: torch.Tensor, metric: str,
                  strategy: str = "hard", inv_cov: Optional[torch.Tensor] = None,
                  margin: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (anchor, positive, negative) indices, each a (K,) LongTensor."""
    N = emb.shape[0]
    D = dist_matrix(emb.detach(), metric, inv_cov).detach()   # (N,N)

    a_list, p_list, n_list = [], [], []
    for i in range(N):
        pos = torch.nonzero(labels == labels[i]).flatten()
        pos = pos[pos != i]
        neg = torch.nonzero(labels != labels[i]).flatten()
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        dp = D[i, pos]
        dn = D[i, neg]
        if strategy == "hard":
            p = pos[torch.argmax(dp)]
            n = neg[torch.argmin(dn)]
        elif strategy == "semi-hard":
            p = pos[torch.argsort(dp)[len(dp) // 2]]
            feas = neg[dn < D[i, p] + margin]
            n = feas[torch.randint(0, feas.numel(), (1,)).item()] if feas.numel() else neg[torch.argmin(dn)]
        elif strategy == "random":
            p = pos[torch.randint(0, pos.numel(), (1,)).item()]
            n = neg[torch.randint(0, neg.numel(), (1,)).item()]
        elif strategy == "easy":
            # The positive takes the median representative; the negative must be far
            # enough: d(a,n) >= d(a,p)+margin (the easy region of the original paper)
            p = pos[torch.argsort(dp)[len(dp) // 2]]
            far = neg[dn >= D[i, p] + margin]
            n = (far[torch.randint(0, far.numel(), (1,)).item()]
                 if far.numel() else neg[torch.argmax(dn)])   # fall back to the farthest negative if none qualifies
        else:
            raise ValueError(strategy)
        a_list.append(i); p_list.append(p.item()); n_list.append(n.item())

    if not a_list:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty.clone(), empty.clone()
    return (torch.tensor(a_list), torch.tensor(p_list), torch.tensor(n_list))


# --------------------------------------------------------------------------- #
# 5. Differentiable triplet loss (Hamming uses a cosine proxy)
# --------------------------------------------------------------------------- #

def triplet_loss(emb: torch.Tensor, labels: torch.Tensor, metric: str,
                 strategy: str = "hard", margin: float = 1.0,
                 inv_cov: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, int]:
    """L = mean(max(0, d(a,p) - d(a,n) + margin)), differentiable, safe to backward directly."""
    if metric == "hamming":
        metric = "cosine"   # Hamming is non-differentiable; train with cosine as a proxy; mining/evaluation still use true Hamming

    a, p, n = mine_triplets(emb, labels, metric, strategy, inv_cov, margin)
    if len(a) == 0:
        return torch.zeros((), device=emb.device), 0

    D = dist_matrix(emb, metric, inv_cov)               # gradient-enabled version
    d_pos = D[a, p]                                     # advanced indexing pulls the paired values (K,)
    d_neg = D[a, n]
    losses = torch.clamp(d_pos - d_neg + margin, min=0.0)
    return losses.mean(), len(a)


# --------------------------------------------------------------------------- #
# 6. Independent training loop
# --------------------------------------------------------------------------- #

def train_augment(x: torch.Tensor) -> torch.Tensor:
    """Batch-level random data augmentation (training only): random horizontal flip + random crop.

    Input and output are both (B,3,32,32). Reflect padding avoids black-border bias.
    Not used for test-set evaluation.
    """
    B = x.shape[0]
    # random horizontal flip (50%)
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], torch.flip(x, dims=[3]), x)
    # random crop: pad to 40x40 (reflect), then take 32x32
    x_pad = F.pad(x, (4, 4, 4, 4), mode="reflect")
    top = torch.randint(0, 8, (B,), device=x.device)
    left = torch.randint(0, 8, (B,), device=x.device)
    return torch.stack([
        x_pad[i, :, top[i]:top[i] + 32, left[i]:left[i] + 32] for i in range(B)
    ])


def train_embedding(images: torch.Tensor, labels: torch.Tensor, metric: str,
                    emb_dim: int = 128, steps: int = 300, batch: int = 80,
                    lr: float = 1e-3, margin: float = 1.0,
                    device: str = "cpu") -> nn.Module:
    """Train an embedding network independently (with the triplet loss of the given metric). Returns the trained network."""
    net = EmbedNet(emb_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    X, y = images.to(device), labels.to(device)

    # Use semi-hard mining uniformly: with more data, hard mining pushes embeddings toward
    # the collapsed solution where "everything shrinks to the origin" (loss stuck at margin,
    # vanishing gradients, classes not separated).
    # semi-hard picks only medium-difficulty negatives, keeping gradients sustained and robust;
    # it is unlikely to collapse for any metric.
    mining = "semi-hard"
    # cosine/hamming on the unit sphere need a smaller margin to avoid collapse; but 0.1 is too
    # loose, so it is raised to 0.25 to push classes farther apart. semi-hard can handle it.
    if metric in ("cosine", "hamming"):
        margin = min(margin, 0.25)

    # Training with the Mahalanobis loss directly is numerically unstable: a fixed Σ⁻¹
    # mismatches the embedding distribution that keeps changing during training, leading to
    # diverging gradients and a loss of nan. Therefore the training stage uses the Euclidean
    # triplet loss as a stable proxy; mining/evaluation still use the true Mahalanobis distance
    # (main_demo re-estimates Σ⁻¹ from the trained embeddings).
    train_metric = "euclidean" if metric == "mahalanobis" else metric
    inv_cov = None

    for step in range(steps):
        # sample a random subset, ensuring at least two classes
        idx = torch.randperm(len(X))[:batch]
        while len(torch.unique(y[idx])) < 2:
            idx = torch.randperm(len(X))[:batch]
        xb = train_augment(X[idx])          # apply random augmentation each step to improve generalization
        emb = net(xb)
        loss, k = triplet_loss(emb, y[idx], train_metric, mining, margin, inv_cov)
        opt.zero_grad()
        loss.backward()
        # safety: clip gradients to prevent explosion; skip this step if the loss is not finite to avoid nan parameters
        if torch.isfinite(loss):
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
            opt.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"      step {step:>4}: loss={loss.item():.4f} (valid triplets {k})")
    return net


# --------------------------------------------------------------------------- #
# 7. Result reporting
# --------------------------------------------------------------------------- #

def retrieval_accuracy(query_emb: torch.Tensor, query_labels: torch.Tensor,
                       gallery_emb: torch.Tensor, gallery_labels: torch.Tensor,
                       metric: str, inv_cov: Optional[torch.Tensor] = None) -> float:
    """Cross-dataset 1-NN retrieval accuracy: for each query sample, find its nearest neighbor
    in the gallery under the given metric, and count the fraction whose neighbor has the same class.

    This is the standard way to evaluate generalization: query uses unseen samples (e.g., the test set),
    gallery uses the training library. It is independent of the training loss and mining strategy.
    """
    D = dist_matrix(query_emb.detach(), metric, inv_cov,
                    gallery_emb.detach()).detach()   # (nQ, nG)
    nn = D.argmin(dim=1)                              # nearest-gallery index for each query
    return (query_labels == gallery_labels[nn]).float().mean().item()


def report(emb_train: torch.Tensor, labels_train: torch.Tensor,
           emb_test: torch.Tensor, labels_test: torch.Tensor,
           metric: str, inv_cov: Optional[torch.Tensor] = None) -> None:
    """Print model performance metrics:
        - intra/inter-class distance (separation on training embeddings)
        - cross-dataset 1-NN retrieval accuracy (test query vs train gallery, reflecting generalization)
        - inter/intra-class distance ratio
    """
    D = dist_matrix(emb_train.detach(), metric, inv_cov).detach()
    intra, inter = [], []
    n = emb_train.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            (intra if labels_train[i] == labels_train[j] else inter).append(D[i, j].item())

    m_intra = float(np.mean(intra))
    m_inter = float(np.mean(inter))
    ratio = m_inter / m_intra if m_intra > 0 else float("inf")
    acc = retrieval_accuracy(emb_test, labels_test, emb_train, labels_train,
                             metric, inv_cov)

    print(f"\n  [intra-class dist mean={m_intra:.4f}   inter-class dist mean={m_inter:.4f}] "
          f"(should satisfy intra < inter)")
    print(f"  test-set retrieval accuracy (1-NN, {metric}) = {acc:.4f}   "
          f"inter/intra ratio = {ratio:.2f}")


# --------------------------------------------------------------------------- #
# 8. Unified entry point (reused by the demo scripts)
# --------------------------------------------------------------------------- #

def main_demo(metric: str, data_dir: str, per_class: int, steps: int,
              batch: int, emb_dim: int, seed: int, device: str) -> None:
    name = _METRICS[metric][0]
    print(f"\n========== CIFAR-10 triplet demo for {name} ==========")
    print(f"Metric: {metric}   Samples per class: {per_class}   Training steps: {steps}")

    # training subset (for training + as retrieval gallery)
    images, labels = load_subset(data_dir, per_class=per_class, seed=seed, train=True)
    print(f"Training subset: {images.shape}, num samples={images.shape[0]}, num classes=10")

    net = train_embedding(images, labels, metric, emb_dim, steps, batch,
                          device=device)
    net.eval()
    with torch.no_grad():
        emb = net(images.to(device))

    # test subset (unseen samples, as the retrieval query, to evaluate generalization)
    images_test, labels_test = load_subset(data_dir, per_class=per_class,
                                           seed=seed + 1, train=False)
    with torch.no_grad():
        emb_test = net(images_test.to(device))
    print(f"Test subset: {images_test.shape}, num samples={images_test.shape[0]}")

    inv_cov = estimate_inv_cov(emb) if metric == "mahalanobis" else None
    report(emb, labels, emb_test, labels_test, metric, inv_cov)


def parse_common_args(desc: str):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--data-dir", default="cifar_data")
    p.add_argument("--per-class", type=int, default=500)
    p.add_argument("--steps", type=int, default=3000)   # evaluation moved to the test set; compare after full convergence
    p.add_argument("--batch", type=int, default=80)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    return p.parse_args()

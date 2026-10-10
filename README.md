# Triplet Loss with Multiple Distance Metrics (CIFAR-10)

Implement positive/negative mining for triplet loss using **7 distance metrics**. Each metric is an independent trainable demo on CIFAR-10, and the metrics are trained separately, with **test-set 1-NN retrieval accuracy** as the core metric.

## Features

- 7 distance metrics: `cosine` (余弦) / `euclidean` (欧氏) / `manhattan` (曼哈顿) / `chebyshev` (切比雪夫) / `minkowski` (闵可夫斯基) / `hamming` (汉明) / `mahalanobis` (马氏)
- One independent script per metric; `semi-hard` mining, batch-level data augmentation, `AdamW` + cosine annealing
- Cross-dataset evaluation (test-set query vs training-set gallery)
- Built-in CIFAR-10 auto-download; a local copy also works
- Includes a detailed explanation of the triplet loss principle (see below)

## Triplet Loss Principle

The core of **triplet loss** is a triplet `(anchor, positive, negative)`:

| Role | Symbol | Meaning |
|---|---|---|
| anchor | `a` | the reference sample |
| positive | `p` | a sample of the **same class** as the anchor |
| negative | `n` | a sample of a **different class** from the anchor |

**Goal**: keep the anchor–positive distance `d(a,p)` small and the anchor–negative distance `d(a,n)` large, separated by at least `margin`. The loss is:

```
L = max( d(a,p) - d(a,n) + margin, 0 )
```

**When gradients are produced**:
- When `d(a,n)` is already larger than `d(a,p) + margin`, the loss is 0 and the triplet **does not update gradients**;
- Only when the triplet "violates the margin" (i.e., `d(a,p) - d(a,n) + margin > 0`) does it produce gradients, pulling the positive closer and pushing the negative away.

**Positive/negative mining** decides which negative is used to form the triplet. There are three classic strategies:

| Strategy | Selected negative | Characteristics |
|---|---|---|
| **hard** | the negative with the smallest `d(a,n)` (hardest) | large gradients, but prone to collapse with large data (embeddings shrink to the origin) |
| **semi-hard** | a negative with `d(a,n)` in `[d(a,p), d(a,p)+margin)` | medium difficulty, sustained and robust gradients |
| **easy** | a negative with `d(a,n) ≥ d(a,p)+margin` | does not violate the margin; loss=0, does not participate in training |

**This project defaults to semi-hard**. The distance `d` can be computed with many metrics; this demo makes the metric a switchable argument to compare training and retrieval across metrics.

## Requirements

```text
Python 3.8+
torch (CPU version is enough)
torchvision
numpy
```

## Quick Start

```bash
# cosine demo (default arguments)
python demo_cosine.py

# specify samples per class and steps
python demo_euclidean.py   --per-class 500 --steps 3000
python demo_mahalanobis.py --per-class 500 --steps 3000
```

On the first run, CIFAR-10 is downloaded automatically to `cifar_data/`; you may also copy the official `cifar-10-batches-py` into that directory, and the download will be skipped.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `cifar_data` | CIFAR-10 data root directory |
| `--per-class` | 500 | number of samples per class (one for training, one for testing) |
| `--steps` | 3000 | number of training steps |
| `--batch` | 80 | batch size |
| `--emb-dim` | 128 | embedding dimension |
| `--seed` | 0 | random seed (reproducible) |
| `--device` | cpu | training device |

## Project Structure

```text
.
├── cifar_triplet_utils.py      # core library: data / distance matrix / mining / loss / augmentation / training / evaluation
├── demo_cosine.py              # cosine similarity demo (余弦相似度)
├── demo_euclidean.py           # euclidean distance demo (欧氏距离)
├── demo_manhattan.py           # manhattan (L1) distance demo (曼哈顿距离)
├── demo_chebyshev.py           # chebyshev (L∞) distance demo (切比雪夫距离)
├── demo_minkowski.py           # minkowski distance demo (闵可夫斯基距离)
├── demo_hamming.py             # hamming distance demo (汉明距离)
├── demo_mahalanobis.py         # mahalanobis distance demo (马氏距离)
├── README.md
└── LICENSE
```

## Design Notes

- **`semi-hard` mining**: semi-hard picks only medium-difficulty negatives, keeping gradients sustained and robust.
- **`easy` strategy**: negatives satisfying `d(a,n) ≥ d(a,p) + margin` participate in mining but not in training (loss is always 0).
- **Data augmentation**: random horizontal flip + random crop (reflect padding) per batch; the test set is not augmented.
- **Optimizer**: `AdamW` + cosine annealing `CosineAnnealingLR`.
- **Two proxy trainings** (stated as-is):
  - Mahalanobis (马氏): trained with a Euclidean proxy; mining/evaluation use the true Mahalanobis distance (direct Mahalanobis training is unstable — the embedding distribution mismatches the fixed Σ⁻¹);
  - Hamming (汉明): trained with a cosine proxy; evaluation uses sign binarization.
- **Evaluation metrics**: intra/inter-class distance + test-set 1-NN retrieval accuracy.

## Results (per-class=500, steps=3000)

| Distance | Test-set 1-NN |
|---|---|
| cosine (余弦) | ✅ > 0.6 |
| euclidean (欧氏) | ✅ > 0.6 |
| minkowski (闵可夫斯基, p=3) | ✅ > 0.6 |
| hamming (汉明) | ✅ > 0.6 |
| manhattan (曼哈顿) | ❌ < 0.6 |
| chebyshev (切比雪夫) | ❌ < 0.6 |
| mahalanobis (马氏) | ❌ < 0.6 |

> The high/low differences come from each metric's own retrieval characteristics and the "training/evaluation metric consistency", not from training quality (they share the same network and the same arguments).

## Open Questions — Help Wanted

The 1-NN retrieval accuracy of the current model on CIFAR-10 still has room for improvement. We look forward to developers sharing optimization ideas and implementations:

1. **Why manhattan / chebyshev / mahalanobis fall below 0.6**
   - manhattan (曼哈顿): L1 is non-differentiable at 0 and is sensitive to linear accumulation of high-dimensional noise;
   - chebyshev (切比雪夫): only takes the single-dimension max, so the retrieval information utilization is the lowest;
   - mahalanobis (马氏): the training (Euclidean proxy) and evaluation (true Mahalanobis) metrics are inconsistent.
   - Possible optimizations: Huber-smoothed L1, hybrid metrics, per-class local covariance, differentiable Mahalanobis training, etc.
2. **How to make Mahalanobis truly participate in end-to-end training** (rather than "Euclidean-proxy training + true Mahalanobis evaluation").
3. **Breaking the accuracy ceiling**: small CNN + semi-hard + fixed steps. Directions: larger backbone, N-pair / lifted hard-negative mining, dynamic margin, longer training.
4. **GPU acceleration**: currently CPU-only torch, which is relatively slow.

## Contributing

- Open an issue: label the change direction (model / training / mining strategy / metric).
- Submit a PR: please include the comparison experiment on CIFAR-10 under the same configuration (`--per-class 500 --steps 3000`).
- Contributions are welcome for any of the "Open Questions" above.

## License

[MIT](./LICENSE)

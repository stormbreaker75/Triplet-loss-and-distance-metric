"""CIFAR-10 Triplet positive and negative examples demo  —— 汉明距离 (hamming)。

Explanation: Hamming binarization is non-differentiable.
The cosine triplet loss is used as a smoothing proxy for training;
mining and evaluation use the true Hamming distance (input embedding > 0.5 for binarization).
"""
import cifar_triplet_utils as U

if __name__ == "__main__":
    a = U.parse_common_args("汉明距离(hamming) ")
    U.main_demo("hamming", a.data_dir, a.per_class, a.steps, a.batch,
                a.emb_dim, a.seed, a.device)

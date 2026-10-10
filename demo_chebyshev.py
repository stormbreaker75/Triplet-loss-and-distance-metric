"""CIFAR-10 Triplet positive and negative examples demo —— 切比雪夫距离 (chebyshev)。"""
import cifar_triplet_utils as U

if __name__ == "__main__":
    a = U.parse_common_args("切比雪夫距离 ")
    U.main_demo("chebyshev", a.data_dir, a.per_class, a.steps, a.batch,
                a.emb_dim, a.seed, a.device)

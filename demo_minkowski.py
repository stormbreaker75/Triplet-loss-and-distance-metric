"""CIFAR-10 Triplet positive and negative examples demo —— 闵可夫斯基距离 (minkowski, p=3)。"""
import cifar_triplet_utils as U

if __name__ == "__main__":
    a = U.parse_common_args("闵可夫斯基距离(p=3) ")
    U.main_demo("minkowski", a.data_dir, a.per_class, a.steps, a.batch,
                a.emb_dim, a.seed, a.device)

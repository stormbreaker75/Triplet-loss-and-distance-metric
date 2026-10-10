"""CIFAR-10 Triplet positive and negative examples demo —— 欧氏距离 (euclidean)。"""
import cifar_triplet_utils as U

if __name__ == "__main__":
    a = U.parse_common_args("欧氏距离(euclidean) ")
    U.main_demo("euclidean", a.data_dir, a.per_class, a.steps, a.batch,
                a.emb_dim, a.seed, a.device)

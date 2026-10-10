"""CIFAR-10 Triplet positive and negative examples demo —— 曼哈顿距离 (manhattan)。"""
import cifar_triplet_utils as U

if __name__ == "__main__":
    a = U.parse_common_args("曼哈顿距离 ")
    U.main_demo("manhattan", a.data_dir, a.per_class, a.steps, a.batch,
                a.emb_dim, a.seed, a.device)

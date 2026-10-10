"""CIFAR-10 Triplet positive and negative examples demo —— 余弦相似度 (cosine)。"""
import cifar_triplet_utils as U

if __name__ == "__main__":
    a = U.parse_common_args("余弦相似度(cosine) ")
    U.main_demo("cosine", a.data_dir, a.per_class, a.steps, a.batch,
                a.emb_dim, a.seed, a.device)

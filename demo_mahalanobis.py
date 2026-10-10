"""CIFAR-10 Triplet positive and negative examples demo —— 马氏距离 (mahalanobis)。


Explanation: The Mahalanobis distance has extremely unstable training loss values
when using a fixed covariance inverse matrix (it may diverge to nan).
Therefore, during the training phase, the Euclidean triplet loss is used as a stable proxy;
in the mining/evaluation phase, the covariance inverse matrix after training is re-evaluated,
and positive and negative examples are selected based on the true Mahalanobis distance.

"""
import cifar_triplet_utils as U

if __name__ == "__main__":
    a = U.parse_common_args("马氏距离 ")
    U.main_demo("mahalanobis", a.data_dir, a.per_class, a.steps, a.batch,
                a.emb_dim, a.seed, a.device)

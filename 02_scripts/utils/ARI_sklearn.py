import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, rand_score


def order_pairs(A, B):
    """
    Reorder the rows and columns of A to minimize pairwise error between A and B.
    """
    # Compute the pairwise distance matrix for rows
    D = np.linalg.norm(A[:, np.newaxis, :] - B[np.newaxis, :, :], axis=2)

    # Reorder A according to the optimal row assignment
    A_tilde = A

    # Now reorder columns of A_tilde to minimize column-wise error
    D_columns = np.linalg.norm(A_tilde[np.newaxis, :, :] - B[:, np.newaxis, :], axis=0)
    _, col_order = linear_sum_assignment(D_columns)
    print(col_order)

    # Reorder columns
    A_tilde = A[:, col_order]

    return A_tilde

C = np.array(pd.read_csv("01_real_data/C_admm_bestParams.csv", sep=",", header=None))
C_GT = np.array(pd.read_csv("C_true_MOB.csv", sep=",", header=None), dtype=int)
print(sum(C_GT))

C_admm = np.zeros_like(C)
max_indices = np.argmax(C, axis=1)
C_admm[np.arange(C.shape[0]), max_indices] = 1

C_admm2 = order_pairs(C_admm, C_GT)
print(sum(C_admm2))
C_admm2 = C_admm2[:, [0,3,1,2,4]]

admm_lab = np.argmax(C_admm, axis=1)
admm2_lab = np.argmax(C_admm2, axis=1)

GT_lab = np.argmax(C_GT, axis=1)

print(admm_lab)
print(admm2_lab)
print(GT_lab)


# Calculate Adjusted Rand Index
ari = adjusted_rand_score(admm_lab, GT_lab)
print("Adjusted Rand Index (ARI):", ari)

ri = rand_score(admm_lab, GT_lab)
print("Rand Index (RI):", ri)

ari = adjusted_rand_score(admm2_lab, GT_lab)
print("Adjusted Rand Index (ARI):", ari)

ri = rand_score(admm2_lab, GT_lab)
print("Rand Index (RI):", ri)

# np.savetxt("C_GT_ARI_testing.csv", GT_lab, delimiter=",", fmt="%d")
# np.savetxt("C_admm_ARI_testing.csv", admm_lab, delimiter=",", fmt="%d")


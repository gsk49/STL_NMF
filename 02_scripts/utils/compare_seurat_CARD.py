import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


c_ref = pd.read_csv("01_real_data/SEURAT_MOB_louvain.csv", header=0, sep=",")
c_ref = (np.array(c_ref)[:,1]).T



### Change Vector to Matrix for multiplication

## Use this for Louvain, ML Refinement and SLM
C = np.zeros((c_ref.size, c_ref.max()+1))
C[np.arange(c_ref.size), c_ref] = 1
## Use this for Leiden
# C = np.zeros((c_ref.size, c_ref.max()))
# C[np.arange(c_ref.size), c_ref-1] = 1

### Load in "Ground Truth" C result to ensure columns align
real_C = np.array(pd.read_csv("01_real_data/C_GSE121891_Figure_2_metadata_filtered_sort.csv", header=0, sep=","))[:,1]
real_C_idx = []
for i in real_C:
    if i == "EPL-IN":
        real_C_idx.append(0)
    elif i == "GC":
        real_C_idx.append(1)
    elif i == "M/TC":
        real_C_idx.append(2)
    elif i == "OSs":
        real_C_idx.append(3)
    elif i == "PGC":
        real_C_idx.append(4)


real_C_idx = np.array(real_C_idx, dtype=int)
C2 = np.zeros((real_C_idx.size, real_C_idx.max()+1))
C2[np.arange(real_C_idx.size), real_C_idx] = 1


### sees which cols correspond to which cols
distance_matrix = cdist(C.T, C2.T, 'euclidean')
row_ind, col_ind = linear_sum_assignment(distance_matrix)
# for i in range(len(row_ind)):
    # print(f"Column {row_ind[i]} of Matrix C is matched with Column {col_ind[i]} of Matrix C2")

### Reorders clusters to be aligned
df = pd.DataFrame({'x': col_ind, 'y': row_ind})
df = df.sort_values('x')
df = np.array(df)
df = df[:, 1]

C = C[:, df]

### Creates N
n = np.identity(len(C[0]))
n2 = np.identity(len(C[0]))
for i in range(len(C[0])):
    n[i,i] /= sum(C[:,i])
    n2[i,i] /= sum(C2[:,i])


### Load in S
S = pd.read_csv("01_real_data/S_OB_6_runs_processed_seurat.dge_filtered_select.csv", header=0, sep=",")
S = np.array(S)[:,1:]
S = np.array(S, dtype=float)

### For tests with GT
real_B = np.matmul(np.matmul(S,C2),n2)
np.savetxt("B_OB.csv", real_B, fmt="%f", delimiter=",")

seurat_B = np.matmul(np.matmul(S,C),n)

print(np.linalg.norm(seurat_B-real_B)/np.linalg.norm(real_B))


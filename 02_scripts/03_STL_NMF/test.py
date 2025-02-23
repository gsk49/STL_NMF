import torch
import pandas as pd
import numpy as np
from deepNMF import *
import os


def get_corr_files(files):
    parts = files.split('_')
    key_part = '_'.join(parts[:2])
    V = 'v' + key_part[1:] + ".csv"
    I = 'i' + key_part[1:] + ".csv"
    # X = 'x' + key_part[1:] + "_0_sim.csv"
    if key_part[-1] == '0':
        t = 0
    elif key_part[-1] == '1':
        t = 1
    else:
        t = 2
    return V, I, t

def get_data(x_file):
    inputs = []
    V_name, I_name, _ = get_corr_files(x_file)

    X = pd.read_csv('./00_synthetic/00_PDAC_A/00_No_X_Noise/05_clust/01_x/' + x_file, sep=",", header=None)
    X = torch.tensor(np.array(X), dtype=torch.float32, device=device)
    
    V = pd.read_csv("./00_synthetic/00_PDAC_A/00_No_X_Noise/05_clust/02_v/v/" + V_name, sep=" ", header=None)
    V = torch.tensor(np.array(V), dtype=torch.float32, device=device)
    
    I = pd.read_csv("./00_synthetic/00_PDAC_A/00_No_X_Noise/03_clust/02_v/intensity/" + I_name, sep=" ", header=None)
    I = torch.tensor(np.array(I), dtype=torch.float32, device=device)
    
    inputs.append((X, V, I))

    return inputs
device="mps"



B = pd.read_csv("./03_results/b_pdac_03clust.csv", header=None)
B = torch.tensor(np.array(B), dtype=torch.float32, device=device)
locs = pd.read_csv("./01_real_data/locs.csv", header=None)
locs = torch.tensor(np.array(locs), dtype=torch.float32, device=device)

X = pd.read_csv("./01_real_data/pdac_a.tsv", sep="\t").set_index("Genes")
row_names = pd.read_csv("./Genes_titled.csv", sep=" ", header=0).set_index("Genes")

X = X[X.index.isin(row_names.index)]
X = np.array(X.sort_index())
# np.savetxt("X_real.csv", X, delimiter=",", fmt="%d")
X = np.array(pd.read_csv("./X_real.tsv", sep="\t", index_col=0, header=0))
X = torch.tensor(X, dtype=torch.float32, device=device)

inputs = get_data('x0_0_0_sim.csv')
_, _, I = inputs[0][0], inputs[0][1], inputs[0][2]
V = torch.ones((len(B[0]), len(X[0])), device="mps")
V /= len(X[0])
model = SuperNet(n_layers=150, locs=locs, B=B, I=I)
model.to(device=device)
model.load_state_dict(torch.load(os.getcwd()+"/model_3.pth"))
model.eval()

out, _ = model(X, V)

np.savetxt("V_group_sparsity.csv", out.cpu().detach().numpy(), delimiter=',')


# X = X[X.index.isin(row_names.index)]
# X = np.array(X.sort_index())
# # np.savetxt("X_real.csv", X, delimiter=",", fmt="%d")
# X = np.array(pd.read_csv("./X_real.tsv", sep="\t", index_col=0, header=0))
# X = torch.tensor(X, dtype=torch.float32, device=device)



# x_names = np.array(pd.read_csv("./xfiles.csv", header=None, sep=","))
# inputs = get_data(x_names[0][0])
# _, _, I = inputs[0][0], inputs[0][1], inputs[0][2]

# for name in x_names:
#     X = torch.tensor(np.array(pd.read_csv("00_synthetic/00_PDAC_A/01_X_Noise/05_clust/01_x/"+str(name[0]), sep=",", header=None)), dtype=torch.float32, device=device)

#     V = torch.ones((len(B[0]), len(X[0])), device="mps")
#     V /= len(X[0])

#     model = SuperNet(n_layers=150, locs=locs, B=B, I=I)
#     model.to(device=device)
#     model.load_state_dict(torch.load(os.getcwd()+"/model_5.pth"))
#     model.eval()



#     out, _ = model(X, V)
#     break
#     np.savetxt("./03_results/noisy/5clust/DEEPNMF_"+str(name[0]), out.cpu().detach().numpy(), delimiter=',')

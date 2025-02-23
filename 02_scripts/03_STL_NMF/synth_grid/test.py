import torch
import pandas as pd
import numpy as np
from deepNMF import *
import os

device="mps"



B = pd.read_csv("./00_synthetic/ruitao/square_2x2/B.csv", header=0, index_col=0)
B = torch.tensor(np.array(B), dtype=torch.float32, device=device)

rV = pd.read_csv("./00_synthetic/ruitao/square_2x2/V.csv", header=0, index_col=0)
rV = torch.tensor(np.array(rV), dtype=torch.float32, device=device).T

X = torch.matmul(B,rV)

locs = pd.read_csv("./00_synthetic/ruitao/square_2x2/regions.csv", header=None)
locs = torch.tensor(np.array(locs).T[0:-1], dtype=torch.float32, device=device).T

I = torch.tensor(np.array(pd.read_csv("00_synthetic/ruitao/square_2x2/i.csv", sep=",", header=None)), dtype=torch.float32, device=device)

V = torch.ones((len(B[0]), len(X[0])), device="mps")
V /= len(X[0])

model = SuperNet(n_layers=150, locs=locs, B=B, I=I)
model.to(device=device)
model.load_state_dict(torch.load(os.getcwd()+"/model_synth.pth"))
model.eval()

out, _ = model(X, V)

np.savetxt("V_square_2x2.csv", out.cpu().detach().numpy(), delimiter=',')


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

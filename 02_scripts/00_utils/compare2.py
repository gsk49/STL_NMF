# This script creates a grid of subplots to compare the relative error of the deconvolution methods.

import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



def get_corr_files(files):
    parts = files.split('_')
    key_part = '_'.join(parts[:4])
    V = key_part[5:] + ".csv"
    v_part = V.split("_")
    v_keypart = "_".join(v_part[:2])
    V = "v"+v_keypart[1:]+".csv"
    if v_keypart[-1] == '0':
        t = 0
    elif v_keypart[-1] == '1':
        t = 1
    else:
        t = 2

    return V, t

stereo_error= []
stereo_relError = []
card_error= []
card_relError = []
stlnmf_error= []
stlnmf_relError = []
spot_error= []
spot_relError = []
types = []

# cardVs = os.listdir("../CARD/noiseless_3_v")
stereoV_Ds = next(os.walk('./stereoscope/pdac_a_stereo/01_X_Noise/3ct'))[1]

stereoVs = [ [0]*2 for i in range(len(stereoV_Ds))]
for idx, D in enumerate(stereoV_Ds):
    v = os.listdir("./stereoscope/pdac_a_stereo/01_X_Noise/3ct/"+D)
    stereoVs[idx][1] = v[0]
    stereoVs[idx][0] = D

# print(stereoVs)

for pair in stereoVs:
    name = pair[0]
    types.append(int(name[-7]))

    real = np.array(pd.read_csv("./00_synthetic/00_PDAC_A/01_X_Noise/03_clust/02_v/v/v"+name[1:-6]+".csv", sep=" ", header=None))
    real /= sum(real)
    stereo = (np.array(pd.read_csv("./stereoscope/pdac_a_stereo/01_X_Noise/3ct/"+pair[0]+"/"+pair[1], sep="\t", header=0, index_col=0)))
    stereo = np.c_[stereo[:,0],stereo[:,1],stereo[:,2]].T
    # stereo = np.c_[stereo[:,0],stereo[:,1],stereo[:,7],stereo[:,9],stereo[:,6]].T
    stereo /= sum(stereo)

    spot = np.array(pd.read_csv("./spotlight/noisy3/spotlight_"+str(name)+".tsv", sep="\t", header=0))
    # spot = np.c_[spot[:,0],spot[:,1],spot[:,7],spot[:,9],spot[:,6]].T
    spot = np.c_[spot[:,0],spot[:,1],spot[:,2]].T
    spot /= sum(spot)

    card = np.array(pd.read_csv("./CARD/noisy_3_v/card_"+str(name)+".csv", sep=",", header=0)).T
    card /= sum(card)
    stlnmf = np.array(pd.read_csv("./03_results/noisy/3clust/DEEPNMF_"+str(name)+".csv", sep=",", header=None))
    stlnmf /= sum(stlnmf)

    real = torch.tensor(np.array(real), dtype=torch.float32)
    stereo = torch.tensor(np.array(stereo), dtype=torch.float32)
    card = torch.tensor(np.array(card), dtype=torch.float32)
    stlnmf = torch.tensor(np.array(stlnmf), dtype=torch.float32)
    spot = torch.tensor(np.array(spot), dtype=torch.float32)


    stereo_error.append(torch.norm(stereo - real).item())
    stereo_relError.append((torch.norm(stereo - real)/torch.norm(real)).item())

    card_error.append(torch.norm(card - real).item())
    card_relError.append((torch.norm(card - real)/torch.norm(real)).item())

    stlnmf_error.append(torch.norm(stlnmf - real).item())
    stlnmf_relError.append((torch.norm(stlnmf - real)/torch.norm(real)).item())

    spot_error.append(torch.norm(spot - real).item())
    spot_relError.append((torch.norm(spot - real)/torch.norm(real)).item())

print(sum(stereo_error)/len(stereo_error))
print(sum(stereo_relError)/len(stereo_relError))
print(sum(card_error)/len(card_error))
print(sum(card_relError)/len(card_relError))
print(sum(stlnmf_error)/len(stlnmf_error))
print(sum(stlnmf_relError)/len(stlnmf_relError))
print(sum(spot_error)/len(spot_error))
print(sum(spot_relError)/len(spot_relError))

plt.figure(figsize=(9,6), layout='constrained')


plt.scatter(range(len(stlnmf_error)), stlnmf_relError, c="olive", label="STL NMF")
plt.scatter(range(len(stereo_error)), stereo_relError, c="darkblue", label="Stereoscope")
plt.scatter(range(len(card_error)), card_relError, c="firebrick", label="CARD")
plt.scatter(range(len(spot_error)), spot_relError, c="dodgerblue", label="SPOTlight")

plt.xlabel("Method Comparison", weight="bold")
plt.ylabel("Relative Error", weight="bold")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Deconvolution Method")
# plt.get_layout_engine().set(w_pad=.1, h_pad=.1)
# plt.set_title("Comparison of Methods: Noisy, 5 Cell-types", loc="left",
#           fontstyle='italic', fontsize=15, weight="bold")
plt.title("Comparison of Methods: Noisy, 3 Cell-types", loc="left",
          fontstyle='italic', fontsize=15, weight="bold")

plt.savefig("compare_noisy3.png")
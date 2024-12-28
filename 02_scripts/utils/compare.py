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
types = []

# cardVs = os.listdir("../CARD/noiseless_3_v")
stereoV_Ds = next(os.walk('./stereoscope/pdac_a_stereo/00_No_X_Noise/5ct'))[1]

stereoVs = [ [0]*2 for i in range(len(stereoV_Ds))]
for idx, D in enumerate(stereoV_Ds):
    v = os.listdir("./stereoscope/pdac_a_stereo/00_No_X_Noise/5ct/"+D)
    stereoVs[idx][1] = v[0]
    stereoVs[idx][0] = D

# print(stereoVs)

for pair in stereoVs:
    name = pair[0]
    types.append(int(name[-7]))

    real = np.array(pd.read_csv("./00_synthetic/00_PDAC_A/00_No_X_Noise/05_clust/02_v/v/v"+name[1:-6]+".csv", sep=" ", header=None))
    real /= sum(real)
    stereo = (np.array(pd.read_csv("./stereoscope/pdac_a_stereo/00_No_X_Noise/5ct/"+pair[0]+"/"+pair[1], sep="\t", header=0, index_col=0)))
    # stereo = np.c_[stereo[:,0],stereo[:,1],stereo[:,2]].T
    stereo = np.c_[stereo[:,0],stereo[:,1],stereo[:,7],stereo[:,9],stereo[:,6]].T
    stereo /= sum(stereo)

    card = np.array(pd.read_csv("./CARD/noiseless_5_v/card_"+str(name)+".csv", sep=",", header=0)).T
    card /= sum(card)
    stlnmf = np.array(pd.read_csv("./03_results/clean/5clust/DEEPNMF_"+str(name)+".csv", sep=",", header=None))
    stlnmf /= sum(stlnmf)
    real = torch.tensor(np.array(real), dtype=torch.float32)
    stereo = torch.tensor(np.array(stereo), dtype=torch.float32)
    card = torch.tensor(np.array(card), dtype=torch.float32)
    stlnmf = torch.tensor(np.array(stlnmf), dtype=torch.float32)

    stereo_error.append(torch.norm(stereo - real).item())
    stereo_relError.append((torch.norm(stereo - real)/torch.norm(real)).item())

    card_error.append(torch.norm(card - real).item())
    card_relError.append((torch.norm(card - real)/torch.norm(real)).item())

    stlnmf_error.append(torch.norm(stlnmf - real).item())
    stlnmf_relError.append((torch.norm(stlnmf - real)/torch.norm(real)).item())

print(sum(stereo_error)/len(stereo_error))
print(sum(stereo_relError)/len(stereo_relError))
print(sum(card_error)/len(card_error))
print(sum(card_relError)/len(card_relError))
print(sum(stlnmf_error)/len(stlnmf_error))
print(sum(stlnmf_relError)/len(stlnmf_relError))


color_map1 = {0: 'cornflowerblue', 1:"royalblue", 2:"lightsteelblue"}
color_map2 = {0: 'lightcoral', 1:"firebrick", 2:"darkred"}
color_map3 = {0: 'yellowgreen', 1:"darkolivegreen", 2:"olive"}

colors1 = [color_map1[label] for label in types]
colors2 = [color_map2[label] for label in types]
colors3 = [color_map3[label] for label in types]

legend_elements1 = [Line2D([0], [0], marker='o', color='w', label='V0',
                          markerfacecolor='royalblue', markersize=5),Line2D([0], [0], marker='o', color='w', label='V1',
                          markerfacecolor='cornflowerblue', markersize=5), Line2D([0], [0], marker='o', color='w', label='V2',
                          markerfacecolor='lightsteelblue', markersize=5),]
legend_elements2 = [Line2D([0], [0], marker='o', color='w', label='V0',
                          markerfacecolor='firebrick', markersize=5),Line2D([0], [0], marker='o', color='w', label='V1',
                          markerfacecolor='lightcoral', markersize=5), Line2D([0], [0], marker='o', color='w', label='V2',
                          markerfacecolor='darkred', markersize=5),]
legend_elements3 = [Line2D([0], [0], marker='o', color='w', label='V0',
                          markerfacecolor='darkolivegreen', markersize=5),Line2D([0], [0], marker='o', color='w', label='V1',
                          markerfacecolor='yellowgreen', markersize=5), Line2D([0], [0], marker='o', color='w', label='V2',
                          markerfacecolor='olive', markersize=5),]



fig = plt.figure(figsize=(12,8), layout='constrained')

ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)
ax3 = plt.subplot(2,2,3)
ax4 = plt.subplot(2,2,4)

ax1.scatter(range(len(stereo_error)), stereo_error, c=colors1)
ax1.set_ylabel("Absolute Error", weight="bold")
ax1.set_xlabel("Stereoscope", weight="bold")
ax1.legend(handles=legend_elements1, loc='upper left', bbox_to_anchor=(1,1), title="Stereoscope")

ax2.scatter(range(len(card_error)), card_error, c=colors2)
ax2.set_xlabel("CARD", weight="bold")
ax2.legend(handles=legend_elements2, loc='upper left', bbox_to_anchor=(1,1), title="CARD")

ax3.scatter(range(len(stlnmf_error)), stlnmf_error, c=colors3)
ax3.set_ylabel("Absolute Error", weight="bold")
ax3.set_xlabel("STL NMF", weight="bold")
ax3.legend(handles=legend_elements3, loc='upper left', bbox_to_anchor=(1,1), title="STL NMF")

ax4.scatter(range(len(stereo_error)), stereo_error, c="cornflowerblue", label="Stereoscope")
ax4.scatter(range(len(card_error)), card_error, c="firebrick", label="CARD")
ax4.scatter(range(len(stlnmf_error)), stlnmf_error, c="olive", label="STL NMF")
ax4.set_xlabel("Method Comparison", weight="bold")
ax4.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Deconvolution Method")
fig.align_labels()
fig.get_layout_engine().set(w_pad=.1, h_pad=.1)
ax1.set_title("Comparison of Methods: Clean, 5 Cell-types", loc="left",
          fontstyle='italic', fontsize=15, weight="bold")

fig.savefig("compare_clean5.png")


# print(cardVs)
# for v in cardVs:
#     name, t = get_corr_files(v)
#     types.append(t)

#     cardV = np.array(pd.read_csv("../CARD/noiseless_3_v/"+v, sep=",", header=0)).T
#     realV = np.array(pd.read_csv("./00_synthetic/00_PDAC_A/00_No_X_Noise/03_clust/02_v/v/"+name, sep=" ", header=None))

#     cardV = torch.tensor(np.array(cardV), dtype=torch.float32)
#     realV = torch.tensor(np.array(realV), dtype=torch.float32)

#     abs_loss.append(torch.norm(cardV - realV).item())
#     rel_loss.append((torch.norm(cardV - realV)/torch.norm(realV)).item())

# print(sum(abs_loss)/len(abs_loss))
# print(sum(rel_loss)/len(rel_loss))
# print(np.std(abs_loss))
# print(np.std(rel_loss))
# print(abs_loss)

# color_map = {0: 'C0', 1: 'C1', 2: 'C2'}
# colors = [color_map[label] for label in types]



# plt.scatter(range(len(abs_loss)), abs_loss, c=colors)
# plt.show()

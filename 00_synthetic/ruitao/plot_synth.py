import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LogNorm, ListedColormap
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable



stlnmfV = np.array(pd.read_csv("V_square_2x2.csv", sep=",", header=None))
stlnmfV /= sum(stlnmfV)
realV = np.array(pd.read_csv("./00_synthetic/ruitao/square_2x2/V.csv", sep=",", header=0, index_col=0)).T
realV /= sum(realV)

loss_stlnmfV = torch.tensor(np.array(stlnmfV), dtype=torch.float32)
loss_realV = torch.tensor(np.array(realV), dtype=torch.float32)


print("STL NMF:")
print(torch.norm(loss_stlnmfV - loss_realV).item())
print((torch.norm(loss_stlnmfV - loss_realV)/torch.norm(loss_realV)).item())

locs = np.array(pd.read_csv("./00_synthetic/ruitao/square_2x2/regions.csv", sep=",", header=None))


stlnmf_ct1 = np.zeros((16,16))
stlnmf_ct2 = np.zeros((16,16))
stlnmf_ct3 = np.zeros((16,16))
stlnmf_ct4 = np.zeros((16,16))

real_ct1 = np.zeros((16,16))
real_ct2 = np.zeros((16,16))
real_ct3 = np.zeros((16,16))
real_ct4 = np.zeros((16,16))

for _ in range(len(stlnmfV)):
    for j in range(len(stlnmfV[0])):
        stlnmf_ct1[locs[j][1]-1][locs[j][0]-1] = stlnmfV[0][j]
        stlnmf_ct2[locs[j][1]-1][locs[j][0]-1] = stlnmfV[1][j]
        stlnmf_ct3[locs[j][1]-1][locs[j][0]-1] = stlnmfV[2][j]
        stlnmf_ct4[locs[j][1]-1][locs[j][0]-1] = stlnmfV[3][j]

        real_ct1[locs[j][1]-1][locs[j][0]-1] = realV[0][j]
        real_ct2[locs[j][1]-1][locs[j][0]-1] = realV[1][j]
        real_ct3[locs[j][1]-1][locs[j][0]-1] = realV[2][j]
        real_ct4[locs[j][1]-1][locs[j][0]-1] = realV[3][j]



original_cmap = plt.get_cmap("Reds")
colors = original_cmap(np.arange(original_cmap.N))
colors[0] = [1, 1, 1, 1]  # Set the first color to white
custom_cmap = ListedColormap(colors)

fig = plt.figure(figsize=(5,8), layout='constrained')

ax1 = plt.subplot(4,2,1)
ax2 = plt.subplot(4,2,3)
ax3 = plt.subplot(4,2,5)
ax4 = plt.subplot(4,2,7)

ax5 = plt.subplot(4,2,2)
ax6 = plt.subplot(4,2,4)
ax7 = plt.subplot(4,2,6)
ax8 = plt.subplot(4,2,8)

for ax in fig.axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')


im_ct1s = ax5.pcolormesh(stlnmf_ct1, cmap=custom_cmap)
im_ct2s = ax6.pcolormesh(stlnmf_ct2, cmap=custom_cmap)
im_ct3s = ax7.pcolormesh(stlnmf_ct3, cmap=custom_cmap)
im_ct4s = ax8.pcolormesh(stlnmf_ct4, cmap=custom_cmap)

im_ct1r = ax1.pcolormesh(real_ct1, cmap=custom_cmap)
im_ct2r = ax2.pcolormesh(real_ct2, cmap=custom_cmap)
im_ct3r = ax3.pcolormesh(real_ct3, cmap=custom_cmap)
im_ct4r = ax4.pcolormesh(real_ct4, cmap=custom_cmap)



ax1.set_ylabel("Cell Type 1", weight="bold")
ax2.set_ylabel("Cell Type 2", weight="bold")
ax3.set_ylabel("Cell Type 3", weight="bold")
ax4.set_ylabel("Cell Type 4", weight="bold")

ax4.set_xlabel("Real", weight="bold")
ax8.set_xlabel("STL NMF", weight="bold")

ax1.set_title(label="Synthetic Data",
          loc="left",
          fontstyle='italic', fontsize=15, weight="bold")

fig.colorbar(im_ct1s, ax=[ax5, ax6, ax7, ax8], orientation='vertical', fraction=0.3, pad=0.04)
fig.get_layout_engine().set(w_pad=.1, h_pad=.1)
# fig.tight_layout()

fig.savefig("v_square_2x2.png")

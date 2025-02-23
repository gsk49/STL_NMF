import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LogNorm, ListedColormap
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable



stlnmfV = np.array(pd.read_csv("03_results/noisy/clust/DEEPNMF_x0_0_0_sim.csv", sep=",", header=None))
stlnmfV /= sum(stlnmfV)

cardV = np.array(pd.read_csv("./CARD/noiseless_3_v/card_x0_0_0_sim.csv", sep=",", header=0)).T
cardV /= sum(cardV)

stereoV = np.array(pd.read_csv("stereoscope/pdac_a_stereo/00_No_X_Noise/3ct/x0_0_0_sim/W.2024-08-19153302.637385.tsv", sep="\t", header=0, index_col=0))[:,0:3].T
stereoV /= sum(stereoV)

realV = np.array(pd.read_csv("./00_synthetic/00_PDAC_A/01_X_Noise/03_clust/02_v/v/v0_0.csv", sep=" ", header=None))
realV /= sum(realV)

spotV = np.array(pd.read_csv("./spotlight/noisy3/spotlight_x0_0_0_sim.tsv", sep="\t", header=0, index_col=0))[:,0:3].T
spotV /= sum(spotV)

loss_cardV = torch.tensor(np.array(cardV), dtype=torch.float32)
loss_stlnmfV = torch.tensor(np.array(stlnmfV), dtype=torch.float32)
loss_stereoV = torch.tensor(np.array(stereoV), dtype=torch.float32)
loss_realV = torch.tensor(np.array(realV), dtype=torch.float32)
loss_spotV = torch.tensor(np.array(spotV), dtype=torch.float32)



print("CARD:")
print(torch.norm(loss_cardV - loss_realV).item())
print((torch.norm(loss_cardV - loss_realV)/torch.norm(loss_realV)).item())

print("STL NMF:")
print(torch.norm(loss_stlnmfV - loss_realV).item())
print((torch.norm(loss_stlnmfV - loss_realV)/torch.norm(loss_realV)).item())

print("stereo:")
print(torch.norm(loss_stereoV - loss_realV).item())
print((torch.norm(loss_stereoV - loss_realV)/torch.norm(loss_realV)).item())

print("spot:")
print(torch.norm(loss_spotV - loss_realV).item())
print((torch.norm(loss_spotV - loss_realV)/torch.norm(loss_realV)).item())

locs = np.array(pd.read_csv("./01_real_data/locs.csv", sep=",", header=None))


stlnmf_ct1 = np.zeros((22,24))
stlnmf_ct2 = np.zeros((22,24))
stlnmf_ct3 = np.zeros((22,24))

card_ct1 = np.zeros((22,24))
card_ct2 = np.zeros((22,24))
card_ct3 = np.zeros((22,24))

stereo_ct1 = np.zeros((22,24))
stereo_ct2 = np.zeros((22,24))
stereo_ct3 = np.zeros((22,24))

real_ct1 = np.zeros((22,24))
real_ct2 = np.zeros((22,24))
real_ct3 = np.zeros((22,24))

spot_ct1 = np.zeros((22,24))
spot_ct2 = np.zeros((22,24))
spot_ct3 = np.zeros((22,24))

for _ in range(len(stlnmfV)):
    for j in range(len(stlnmfV[0])):
        stlnmf_ct1[locs[j][0]-7][locs[j][1]-7] = stlnmfV[0][j]
        stlnmf_ct2[locs[j][0]-7][locs[j][1]-7] = stlnmfV[1][j]
        stlnmf_ct3[locs[j][0]-7][locs[j][1]-7] = stlnmfV[2][j]

        card_ct1[locs[j][0]-7][locs[j][1]-7] = cardV[0][j]
        card_ct2[locs[j][0]-7][locs[j][1]-7] = cardV[1][j]
        card_ct3[locs[j][0]-7][locs[j][1]-7] = cardV[2][j]

        stereo_ct1[locs[j][0]-7][locs[j][1]-7] = stereoV[0][j]
        stereo_ct2[locs[j][0]-7][locs[j][1]-7] = stereoV[1][j]
        stereo_ct3[locs[j][0]-7][locs[j][1]-7] = stereoV[2][j]

        real_ct1[locs[j][0]-7][locs[j][1]-7] = realV[0][j]
        real_ct2[locs[j][0]-7][locs[j][1]-7] = realV[1][j]
        real_ct3[locs[j][0]-7][locs[j][1]-7] = realV[2][j]

        spot_ct1[locs[j][0]-7][locs[j][1]-7] = spotV[0][j]
        spot_ct2[locs[j][0]-7][locs[j][1]-7] = spotV[1][j]
        spot_ct3[locs[j][0]-7][locs[j][1]-7] = spotV[2][j]


original_cmap = plt.get_cmap("Reds")
colors = original_cmap(np.arange(original_cmap.N))
colors[0] = [1, 1, 1, 1]  # Set the first color to white
custom_cmap = ListedColormap(colors)

fig = plt.figure(figsize=(12,8), layout='constrained')

ax1 = plt.subplot(3,4,4)
ax2 = plt.subplot(3,4,8)
ax3 = plt.subplot(3,4,12)

ax4 = plt.subplot(3,4,2)
ax5 = plt.subplot(3,4,6)
ax6 = plt.subplot(3,4,10)

ax7 = plt.subplot(3,4,1)
ax8 = plt.subplot(3,4,5)
ax9 = plt.subplot(3,4,9)

ax10 = plt.subplot(3,4,3)
ax11 = plt.subplot(3,4,7)
ax12 = plt.subplot(3,4,11)

for ax in fig.axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')


im_ct1s = ax1.pcolormesh(stlnmf_ct1, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct2s = ax2.pcolormesh(stlnmf_ct2, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct3s = ax3.pcolormesh(stlnmf_ct3, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))

im_ct1r = ax4.pcolormesh(card_ct1, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct2r = ax5.pcolormesh(card_ct2, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct3r = ax6.pcolormesh(card_ct3, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))

im_ct1real = ax7.pcolormesh(real_ct1, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct2real = ax8.pcolormesh(real_ct2, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct3real = ax9.pcolormesh(real_ct3, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))

im_ct1stereo = ax10.pcolormesh(stereo_ct1, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct2stereo = ax11.pcolormesh(stereo_ct2, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct3stereo = ax12.pcolormesh(stereo_ct3, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))

ax7.set_ylabel("Cell Type 1: Acinar", weight="bold")
ax8.set_ylabel("Cell Type 2: Cancer Clone A", weight="bold")
ax9.set_ylabel("Cell Type 3: Cancer Clone B", weight="bold")

ax9.set_xlabel("Ground Truth", weight="bold")
ax6.set_xlabel("CARD", weight="bold")
ax12.set_xlabel("Stereoscope", weight="bold")
ax3.set_xlabel("STL NMF", weight="bold")

ax7.set_title(label="Three Clusters, Clean X",
          loc="left",
          fontstyle='italic', fontsize=15, weight="bold")

fig.colorbar(im_ct1s, ax=[ax1, ax2, ax3], orientation='vertical', fraction=0.3, pad=0.04)
fig.get_layout_engine().set(w_pad=.1, h_pad=.1)
# fig.tight_layout()

# fig.savefig("v_clean3_lognorm.png")





fig = plt.figure(figsize=(12,8), layout="constrained")
ax1 = plt.subplot(3,5,4)
ax2 = plt.subplot(3,5,9)
ax3 = plt.subplot(3,5,14)

ax4 = plt.subplot(3,5,2)
ax5 = plt.subplot(3,5,7)
ax6 = plt.subplot(3,5,12)

ax7 = plt.subplot(3,5,1)
ax8 = plt.subplot(3,5,6)
ax9 = plt.subplot(3,5,11)

ax10 = plt.subplot(3,5,3)
ax11 = plt.subplot(3,5,8)
ax12 = plt.subplot(3,5,13)

ax13 = plt.subplot(3,5,5)
ax14 = plt.subplot(3,5,10)
ax15 = plt.subplot(3,5,15)

for ax in fig.axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

im_ct1s = ax1.pcolormesh(stlnmf_ct1, cmap=custom_cmap)
im_ct2s = ax2.pcolormesh(stlnmf_ct2, cmap=custom_cmap)
im_ct3s = ax3.pcolormesh(stlnmf_ct3, cmap=custom_cmap)

im_ct1r = ax4.pcolormesh(card_ct1, cmap=custom_cmap)
im_ct2r = ax5.pcolormesh(card_ct2, cmap=custom_cmap)
im_ct3r = ax6.pcolormesh(card_ct3, cmap=custom_cmap)

im_ct1real = ax7.pcolormesh(real_ct1, cmap=custom_cmap)
im_ct2real = ax8.pcolormesh(real_ct2, cmap=custom_cmap)
im_ct3real = ax9.pcolormesh(real_ct3, cmap=custom_cmap)

im_ct1stereo = ax10.pcolormesh(stereo_ct1, cmap=custom_cmap)
im_ct2stereo = ax11.pcolormesh(stereo_ct2, cmap=custom_cmap)
im_ct3stereo = ax12.pcolormesh(stereo_ct3, cmap=custom_cmap)

im_ct1spot = ax13.pcolormesh(spot_ct1, cmap=custom_cmap)
im_ct2spot = ax14.pcolormesh(spot_ct2, cmap=custom_cmap)
im_ct3spot = ax15.pcolormesh(spot_ct3, cmap=custom_cmap)

ax7.set_ylabel("Cell Type 1: Acinar", weight="bold")
ax8.set_ylabel("Cell Type 2: Cancer Clone A", weight="bold")
ax9.set_ylabel("Cell Type 3: Cancer Clone B", weight="bold")

ax9.set_xlabel("Ground Truth", weight="bold")
ax6.set_xlabel("CARD", weight="bold")
ax12.set_xlabel("Stereoscope", weight="bold")
ax3.set_xlabel("STL NMF", weight="bold")
ax15.set_xlabel("SPOTlight", weight="bold")

ax7.set_title(label="Three Clusters, Noisy X",
          loc="left",
          fontstyle='italic', fontsize=15, weight="bold")

fig.colorbar(im_ct1s, ax=[ax13, ax14, ax15], orientation='vertical', fraction=0.5, pad=0.04)
fig.get_layout_engine().set(w_pad=.1, h_pad=.1)

fig.savefig("v_noisy3.png")

# fig = plt.figure(figsize=(7,10))
# ax1 = plt.subplot(3,2,1)
# ax2 = plt.subplot(3,2,2)

# ax3 = plt.subplot(3,2,3)
# ax4 = plt.subplot(3,2,4)

# ax5 = plt.subplot(3,2,5)
# ax6 = plt.subplot(3,2,6)

# card_ct1 = ax1.pcolormesh(np.abs(r_ct1-real_ct1), cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
# ax1.set_xlabel("CARD, ct1")
# fig.colorbar(card_ct1, ax=ax1)

# STL NMF_ct1 = ax2.pcolormesh(np.abs(s_ct1-real_ct1), cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
# ax2.set_xlabel("STL NMF, ct1")
# fig.colorbar(STL NMF_ct1, ax=ax2)

# card_ct2 = ax3.pcolormesh(np.abs(r_ct2-real_ct2), cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
# ax3.set_xlabel("CARD, ct2")
# fig.colorbar(card_ct2, ax=ax3)

# STL NMF_ct2 = ax4.pcolormesh(np.abs(s_ct2-real_ct2), cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
# ax4.set_xlabel("STL NMF, ct2")
# fig.colorbar(STL NMF_ct2, ax=ax4)

# card_ct3 = ax5.pcolormesh(np.abs(r_ct3-real_ct3), cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
# ax5.set_xlabel("CARD, ct3")
# fig.colorbar(card_ct3, ax=ax5)

# STL NMF_ct3 = ax6.pcolormesh(np.abs(s_ct3-real_ct3), cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
# ax6.set_xlabel("STL NMF, ct3")
# fig.colorbar(STL NMF_ct3, ax=ax6)

# fig.savefig("card_vs_STL NMF_error.png")
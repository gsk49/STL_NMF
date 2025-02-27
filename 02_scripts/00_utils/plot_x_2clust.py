import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LogNorm, ListedColormap
import numpy as np
import torch

# stlnmfV = np.array(pd.read_csv("DEEPNMF_real3.csv", sep=",", header=None))[1:3, :]
stlnmfV = np.array(pd.read_csv("03_results/04_STL_NMF/real/V_real.csv", sep=",", header=None))[1:3,:]
threshold = 1e-3  # Define the threshold value
stlnmfV[stlnmfV < threshold] = 0
stlnmfV /= sum(stlnmfV)+np.ones(len(stlnmfV.T))*.00001

cardV = np.array(pd.read_csv("./03_results/01_CARD/real/card_real_3.csv", sep=",", header=0))[:,1:3].T
cardV /= sum(cardV)+np.ones(len(cardV.T))*.00001

stereoV = np.array(pd.read_csv("03_results/03_stereoscope/real/W.2024-08-21175235.111338.tsv", sep="\t", header=0, index_col=0))[:,1:3].T
stereoV /= sum(stereoV)+np.ones(len(stereoV.T))*.00001

realV = pd.read_csv("./01_real_data/pdac/pdac_a.tsv", sep="\t", header=0)
realV.set_index("Genes", inplace=True)
realV = np.array([realV.loc["TM4SF1"], realV.loc["S100A4"]], dtype=float)
realV /= sum(realV) + np.ones(len(realV.T))*.00001

spotV = np.array(pd.read_csv("03_results/02_spotlight/real/spotlight_real.tsv", sep="\t", header=0, index_col=0))[:,1:3].T
spotV /= sum(spotV)+np.ones(len(spotV.T))*.00001

threshold = 1e-5  # Define the threshold value
#realV[realV < threshold] = 0

# realV /= sum(realV)+np.ones(len(realV.T))*.00001


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

locs = np.array(pd.read_csv("./01_real_data/supplementary/locs.csv", sep=",", header=None))


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

spot_ct3 = np.zeros((22,24))
spot_ct2 = np.zeros((22,24))

for _ in range(len(stlnmfV)):
    for j in range(len(stlnmfV[0])):
        stlnmf_ct2[locs[j][0]-7][locs[j][1]-7] = stlnmfV[0][j]
        stlnmf_ct3[locs[j][0]-7][locs[j][1]-7] = stlnmfV[1][j]

        card_ct2[locs[j][0]-7][locs[j][1]-7] = cardV[0][j]
        card_ct3[locs[j][0]-7][locs[j][1]-7] = cardV[1][j]

        stereo_ct2[locs[j][0]-7][locs[j][1]-7] = stereoV[0][j]
        stereo_ct3[locs[j][0]-7][locs[j][1]-7] = stereoV[1][j]

        real_ct2[locs[j][0]-7][locs[j][1]-7] = realV[0][j]
        real_ct3[locs[j][0]-7][locs[j][1]-7] = realV[1][j]

        spot_ct2[locs[j][0]-7][locs[j][1]-7] = spotV[0][j]
        spot_ct3[locs[j][0]-7][locs[j][1]-7] = spotV[1][j]


original_cmap = plt.get_cmap("Reds")
colors = original_cmap(np.arange(original_cmap.N))
colors[0] = [1, 1, 1, 1]  # Set the first color to white
custom_cmap = ListedColormap(colors)

fig = plt.figure(figsize=(12,8), layout='constrained')

ax1 = plt.subplot(2,5,4)
ax2 = plt.subplot(2,5,9)

ax4 = plt.subplot(2,5,2)
ax5 = plt.subplot(2,5,7)

ax7 = plt.subplot(2,5,1)
ax8 = plt.subplot(2,5,6)

ax10 = plt.subplot(2,5,3)
ax11 = plt.subplot(2,5,8)

ax12 = plt.subplot(2,5,5)
ax13 = plt.subplot(2,5,10)

for ax in fig.axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')


im_ct1s = ax1.pcolormesh(stlnmf_ct2, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct2s = ax2.pcolormesh(stlnmf_ct3, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))

im_ct1r = ax4.pcolormesh(card_ct2, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct2r = ax5.pcolormesh(card_ct3, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))

im_ct1real = ax7.pcolormesh(real_ct2, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct2real = ax8.pcolormesh(real_ct3, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))

im_ct1stereo = ax10.pcolormesh(stereo_ct2, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct2stereo = ax11.pcolormesh(stereo_ct3, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))

im_ct1spot = ax12.pcolormesh(spot_ct2, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))
im_ct2spot = ax13.pcolormesh(spot_ct3, cmap=custom_cmap, norm=LogNorm(vmin=1*np.exp(-10), vmax=1))

ax8.set_ylabel("Cell Type 3: Cancer Clone B", weight="bold")
ax7.set_ylabel("Cell Type 2: Cancer Clone A", weight="bold")

ax8.set_xlabel("Marker Genes", weight="bold")
ax5.set_xlabel("CARD", weight="bold")
ax11.set_xlabel("Stereoscope", weight="bold")
ax2.set_xlabel("STL NMF", weight="bold")
ax13.set_xlabel("SPOTlight", weight="bold")

ax7.set_title(label="Three Clusters, Real Data",
          loc="left",
          fontstyle='italic', fontsize=15, weight="bold")

fig.colorbar(im_ct1s, ax=[ax12, ax13], orientation='vertical', fraction=0.3, pad=0.04)
fig.get_layout_engine().set(w_pad=.1, h_pad=.1)
# fig.tight_layout()

fig.savefig("v_real2_lognorm.png")





fig = plt.figure(figsize=(12,5), layout="constrained")
ax1 = plt.subplot(2,5,1)
ax6 = plt.subplot(2,5,6)

ax2 = plt.subplot(2,5,2)
ax7 = plt.subplot(2,5,7)

ax3 = plt.subplot(2,5,3)
ax8 = plt.subplot(2,5,8)

ax4 = plt.subplot(2,5,4)
ax9 = plt.subplot(2,5,9)

ax5 = plt.subplot(2,5,5)
ax10 = plt.subplot(2,5,10)

for ax in fig.axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

im_ct1real = ax1.pcolormesh(real_ct2, cmap=custom_cmap)
im_ct2real = ax6.pcolormesh(real_ct3, cmap=custom_cmap)

im_ct1s = ax2.pcolormesh(stlnmf_ct2, cmap=custom_cmap)
im_ct2s = ax7.pcolormesh(stlnmf_ct3, cmap=custom_cmap)

im_ct1spot = ax3.pcolormesh(spot_ct2, cmap=custom_cmap)
im_ct2spot = ax8.pcolormesh(spot_ct3, cmap=custom_cmap)

im_ct1stereo = ax5.pcolormesh(stereo_ct2, cmap=custom_cmap)
im_ct2stereo = ax10.pcolormesh(stereo_ct3, cmap=custom_cmap)

im_ct1r = ax4.pcolormesh(card_ct2, cmap=custom_cmap)
im_ct2r = ax9.pcolormesh(card_ct3, cmap=custom_cmap)




ax6.set_ylabel("Cell Type 3: Cancer Clone B", weight="bold")
ax1.set_ylabel("Cell Type 2: Cancer Clone A", weight="bold")

ax6.set_xlabel("Marker Genes", weight="bold")
ax9.set_xlabel("CARD", weight="bold")
ax10.set_xlabel("Stereoscope", weight="bold")
ax7.set_xlabel("STL NMF", weight="bold")
ax8.set_xlabel("SPOTlight", weight="bold")

ax1.set_title(label="Two Clusters, Real Data",
          loc="left",
          fontstyle='italic', fontsize=15, weight="bold")

fig.colorbar(im_ct1s, ax=[ax5, ax10], orientation='vertical', fraction=0.3, pad=0.04)
fig.get_layout_engine().set(w_pad=.1, h_pad=.1)

fig.savefig("v_real2.png")
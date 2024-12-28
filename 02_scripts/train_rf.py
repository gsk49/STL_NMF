import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import random
import os
import math
import pandas as pd
from deepNMF_rf import *
import gc
import matplotlib.pyplot as plt

inf = math.inf
EPSILON = np.finfo(np.float32).eps

class WeightClipper(object):
    def __init__(self, lower=0, upper=inf):
        self.lower = lower
        self.upper = upper

    def __call__(self, module):
        if hasattr(module, "weight"):
            w = module.weight.data
            w = w.clamp(min=self.lower, max=self.upper)
            module.weight.data = w

def divergence(v, V):
    # v, _, _, _, _ = output
    V = V.to(v.device)
    error = torch.norm(v - V)
    error = torch.clamp(error, min=EPSILON)

    return error

def divergence2(v, V, x, X):
    V = V.to(v.device)
    X = X.to(x.device)
    error = torch.norm(v - V) + torch.norm(x - X)/500
    error = torch.clamp(error, min=EPSILON)

    return error

def get_random_files(folder, batch):
    files = os.listdir(folder)
    random_files = random.sample(files, batch)
    return random_files

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

def get_data(batch=64):
    X_names = get_random_files("./00_synthetic/00_PDAC_A/00_No_X_Noise/05_clust/01_x", batch)
    inputs = []

    for x_file in X_names: 

        ### "Synthetic"
        V_name, I_name, t = get_corr_files(x_file)

        X = pd.read_csv('./00_synthetic/00_PDAC_A/00_No_X_Noise/05_clust/01_x/' + x_file, sep=",", header=None)
        X = torch.tensor(np.array(X), dtype=torch.float32, device=device)
        
        V = pd.read_csv("./00_synthetic/00_PDAC_A/00_No_X_Noise/05_clust/02_v/v/" + V_name, sep=" ", header=None)
        V = torch.tensor(np.array(V), dtype=torch.float32, device=device)

        I = pd.read_csv("./00_synthetic/00_PDAC_A/00_No_X_Noise/05_clust/02_v/intensity/"+ I_name, sep=" ", header=None)
        I = torch.tensor(np.array(I), dtype=torch.float32, device=device)
        
        
        inputs.append((X, V, I, t))

    return inputs


    
device = torch.device("mps" if torch.backends.mps.is_available() else 'cpu')
rB = pd.read_csv("./03_results/b_pdac_05clust.csv", header=None)
rB = torch.tensor(np.array(rB), dtype=torch.float32, device=device)
locs = pd.read_csv("./01_real_data/locs.csv", header=None)
locs = torch.tensor(np.array(locs), dtype=torch.float32, device=device)


# constraints = WeightClipper()
model = SuperNet(n_layers=150, locs=locs)
model.to(device)


param_groups = [
    {'params': [p for name, p in model.named_parameters() if "B" in name], 'lr':3}, 
    {'params': [p for name, p in model.named_parameters() if "B" not in name], 'lr':.1}
]
optimizerADAM = optim.Adam(param_groups, weight_decay=.001)
scheduler = StepLR(optimizerADAM, step_size=7, gamma=0.5)


error_X = []
error_X_rel = []
error_V = []
error_V_rel = []
error_B = []
error_B_rel = []

def train():
    types = []
    for i in range(100):

        if True:
            print(str(i)+"\n")      # print epoch

        inputs = get_data(16)       # get 16 random files to train on
        for idx in range(len(inputs)):

            ### get X,I for input and true V for testing -- change _ to t later
            X, rV, I, t = inputs[idx][0], inputs[idx][1], inputs[idx][2], inputs[idx][3]

            ### gen initial V
            V = torch.ones((len(rB[0]), len(X[0])), device=device)
            V /= 5
            V = V + (torch.rand(size=V.shape, device=device))*.5
            
            ### run model
            out, sB = model(X, V, I)

            ### calc synthetic X
            sX = torch.matmul(sB, out)

            ### Calculate error for diff variables
            error_comb = divergence2(out, rV, sX, X)
            error = divergence(sX, X)
            error2 = divergence(out, rV)
            error3 = divergence(sB, rB)

            ### print for debugging -- change to i % 4?
            print("Absolute: "+str(error.item()))
            print("Relative: "+str((error/(torch.norm(X)+EPSILON)).item()))
            
            ### optimize and choose error for training
            optimizerADAM.zero_grad()
            error_comb.backward()
            optimizerADAM.step()

            ### keep B non-negative
            for name, p in model.named_parameters():
                if "B" in name:
                    p.data.clamp_(min=0)
                    p.data.round_()

            ### keep abs and rel error for graphing
            error_X.append(error.item())
            error_X_rel.append((error/(torch.norm(X)+EPSILON)).item())
            error_V.append(error2.item())
            error_V_rel.append((error2/(torch.norm(V)+EPSILON)).item())
            error_B.append(error3.item())
            error_B_rel.append((error3/(torch.norm(rB)+EPSILON)).item())
            types.append(t)         # Change to t once gathering results

        
        ### print param values after each epoch -- reduce later
        for name, param in model.named_parameters():
            print(f'Param: {name}')
            print(f"Val: {param}")

        ### keep memory fresh
        # scheduler.step()
        torch.mps.empty_cache()
        gc.collect()
        print("")

        

        if True:
            ### save model
            torch.save(model.state_dict(), "model.pth")
            ### save B for checking w/ eyes
            np.savetxt("sim_B.csv", sB.cpu().detach().numpy(), delimiter=',')


            # plt.plot(range(len(error_values)), error_values, "o")
            color_map = {0: 'C0', 1: 'C1', 2: 'C2'}
            colors = [color_map[label] for label in types]

            plt.subplot(2,3,1)
            plt.scatter(range(len(error_X)), error_X, c=colors)
            plt.ylabel("Absolute error Values: X")
            plt.xlabel("Epoch #")

            plt.subplot(2,3,4)
            plt.scatter(range(len(error_X_rel)), error_X_rel, c=colors)
            plt.ylabel("Relative error Values: X")
            plt.xlabel("Epoch #")


            plt.subplot(2,3,2)
            plt.scatter(range(len(error_V)), error_V, c=colors)
            plt.ylabel("Abs error Values: V")
            plt.xlabel("Epoch #")

            plt.subplot(2,3,5)
            plt.scatter(range(len(error_V_rel)), error_V_rel, c=colors)
            plt.ylabel("Rel error Values: V")
            plt.xlabel("Epoch #")


            plt.subplot(2,3,3)
            plt.scatter(range(len(error_B)), error_B, c=colors)
            plt.ylabel("Abs error Values: B")
            plt.xlabel("Epoch #")

            plt.subplot(2,3,6)
            plt.scatter(range(len(error_B_rel)), error_B_rel, c=colors)
            plt.ylabel("Rel error Values: B")
            plt.xlabel("Epoch #")

            plt.tight_layout()
            plt.show()


            np.savetxt("real_v.csv", rV.cpu().detach().numpy(), delimiter=',')
            np.savetxt("sim_v.csv", out.cpu().detach().numpy(), delimiter=',')

    print(f'Abs error: {sum(error_X)/len(error_X)}')
    print(f'Rel error: {sum(error_X_rel)/len(error_X_rel)}')
    # np.savetxt("DEEPNMF_v0_0.csv", out.cpu().detach().numpy(), delimiter=',')

torch.mps.empty_cache()
train()

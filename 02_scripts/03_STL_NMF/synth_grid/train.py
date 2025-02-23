import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import random
import os
import math
import pandas as pd
from deepNMF import *
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
    loss = torch.norm(v - V)
    # loss = torch.clamp(loss, min=EPSILON)

    return loss

def get_random_files(folder, batch):
    files = os.listdir(folder)
    random_files = random.sample(files, batch)
    # random_files = files
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
    # X_names = get_random_files("./00_synthetic/00_PDAC_A/02_X_shuf", batch)
    inputs = []

    for x_file in X_names: 
       
        # V_name, I_name, t = get_corr_files(x_file)
        
        V = pd.read_csv("./00_synthetic/ruitao/square_2x2/V.csv", sep=",", header=0, index_col=0)
        V = torch.tensor(np.array(V), dtype=torch.float32, device=device).T

        I = pd.read_csv("./00_synthetic/ruitao/square_2x2/i.csv", sep=",", header=None)
        I = torch.tensor(np.array(I), dtype=torch.float32, device=device)
        
        # print(V)
        
         # X = np.array(X)[:,1:]
        # X = torch.tensor(np.array(X, dtype=np.float32), dtype=torch.float32, device=device)
        
        inputs.append((0, V, I, 0))
    
    # X = pd.read_csv('./00_synthetic/01_x/' + X_names[0], sep=",", header=None)
    # X = torch.tensor(np.array(X), dtype=torch.float32, device=device)
    # V = pd.read_csv("./00_synthetic/02_v/v/" + V_names, sep=" ", header=None)
    # V = torch.tensor(np.array(V), dtype=torch.float32, device=device)
    # I = pd.read_csv("./00_synthetic/02_v/intensity/" + I_names, sep=" ", header=None)
    # I = torch.tensor(np.array(I), dtype=torch.float32, device=device)

    # return [[X], [V], [I]]

    # X = pd.read_csv('./00_synthetic/00_PDAC_A/00_No_X_Noise/03_clust/01_x/' + "x0_0_0_sim.csv", sep=",", header=None)
    # X = torch.tensor(np.array(X), dtype=torch.float32, device=device)
    
    # V = pd.read_csv("./00_synthetic/00_PDAC_A/00_No_X_Noise/03_clust/02_v/v/" + "v0_0.csv", sep=" ", header=None)
    # V = torch.tensor(np.array(V), dtype=torch.float32, device=device)
    
    # I = pd.read_csv("./00_synthetic/00_PDAC_A/00_No_X_Noise/03_clust/02_v/intensity/" + "i0_0.csv", sep=" ", header=None)
    # I = torch.tensor(np.array(I), dtype=torch.float32, device=device)

    # inputs.append((X, V, I))


    return inputs


    
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps" if torch.backends.mps.is_available() else 'cpu')
# B = pd.read_csv("b_pdac_05clust.csv", header=None)
# B = torch.tensor(np.array(B), dtype=torch.float32, device=device)
rB = pd.read_csv("./00_synthetic/ruitao/square_2x2/B.csv", header=0, index_col=0)
rB = torch.tensor(np.array(rB), dtype=torch.float32, device=device)
locs = pd.read_csv("./00_synthetic/ruitao/square_2x2/regions.csv", header=None).T[0:-1].T
print(locs.shape)
locs = torch.tensor(np.array(locs), dtype=torch.float32, device=device)
print(device)

constraints = WeightClipper()

I = pd.read_csv("./00_synthetic/ruitao/square_2x2/i.csv", sep=",", header=None)
I = torch.tensor(np.array(I), dtype=torch.float32, device=device)
model = SuperNet(n_layers=150, locs=locs, B=rB, I=I)

model.to(device)




optimizerADAM = optim.Adam(model.parameters(), lr=.01, weight_decay=.001)
scheduler = StepLR(optimizerADAM, step_size=25, gamma=0.5)

loss_X = []
loss_X2 = []
loss_V = []
loss_V2 = []
types = []
def train():
    
    for i in range(51):

        if True:
            print(str(i)+"\n")

        inputs = get_data(16)
        for idx in range(len(inputs)):
            # print(torch.norm(B-rB))
            _, real_V, I, t = inputs[idx][0], inputs[idx][1], inputs[idx][2], inputs[idx][3]
            types.append(t)
            X = torch.matmul(rB, real_V)
            V = torch.ones((len(rB[0]), len(X[0])), device=device)
            V /= 5
            V = V + (torch.rand(size=V.shape, device=device))*.5

            out, w = model(X, V)
            sX = torch.matmul(rB, out)
            # out = model(X, V, I, B)
            # print(sum(sB))
            # print(sum(rB))
            # np.savetxt("sim_B.csv", sB.cpu().detach().numpy(), delimiter=',')
            # loss = divergence(out, real_V)
            loss = divergence(sX, X)
            loss2 = divergence(out, real_V)

            # print("Absolute: "+str(loss.item()))
            # print("Relative: "+str((loss/(torch.norm(X)+EPSILON)).item()))
            
            optimizerADAM.zero_grad()
            loss2.backward(retain_graph=True)    

            model.apply(constraints)
            
            optimizerADAM.step()

            loss_X.append(loss.item())
            loss_X2.append((loss/(torch.norm(X)+EPSILON)).item())
            loss_V.append(loss2.item())
            loss_V2.append((loss2/(torch.norm(V)+EPSILON)).item())


        # for name, param in model.named_parameters():
        #     print(f'Param: {name}')
        #     print(f"Val: {param}")


        scheduler.step()

        torch.cuda.empty_cache()
        gc.collect()
        print("")

        

        if i%15==0:
            print(w)
            for name, param in model.named_parameters():
                print(f'Param: {name}')
                print(f"Val: {param}")
            torch.save(model.state_dict(), "model_synth.pth")

            color_map = {0: 'C0', 1: 'C1', 2: 'C2'}
            colors = [color_map[label] for label in types]
            plt.scatter(range(len(loss_V2)), loss_V2, c=colors)
            plt.xlabel("Samples")
            plt.ylabel("V Relative Error")
            plt.title("Training of NN Parameters")
            plt.figure()


            plt.subplot(2,2,1)
            plt.scatter(range(len(loss_X)), loss_X)
            plt.ylabel("Absolute Loss Values: X")
            plt.xlabel("Epoch #")

            plt.subplot(2,2,3)
            plt.scatter(range(len(loss_X2)), loss_X2)
            plt.ylabel("Relative Loss Values: X")
            plt.xlabel("Epoch #")


            plt.subplot(2,2,2)
            plt.scatter(range(len(loss_V)), loss_V)
            plt.ylabel("Abs Loss Values: V")
            plt.xlabel("Epoch #")

            plt.subplot(2,2,4)
            plt.scatter(range(len(loss_V2)), loss_V2)
            plt.ylabel("Rel Loss Values: V")
            plt.xlabel("Epoch #")


            plt.tight_layout()
            plt.show()


            # np.savetxt("real_v.csv", real_V.cpu().detach().numpy(), delimiter=',')
            # np.savetxt("sim_v.csv", out.cpu().detach().numpy(), delimiter=',')

    print(f'Abs Loss: {sum(loss_X)/len(loss_X)}')
    print(f'Rel Loss: {sum(loss_X2)/len(loss_X2)}')
    # np.savetxt("DEEPNMF_v0_0.csv", out.cpu().detach().numpy(), delimiter=',')

train()

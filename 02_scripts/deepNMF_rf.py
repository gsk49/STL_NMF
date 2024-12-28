import torch
import torch.nn as nn
import numpy as np

EPSILON = torch.finfo(torch.float32).eps
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# ================================ supervised Net ======================================
class SuperLayer(nn.Module):

    def __init__(self, locs, L1, L2, S1, S2, B1_fc, B2_fc=None):
        super(SuperLayer, self).__init__()

        ### Parameters
        self.l_1 = L1
        self.l_2 = L2
        self.s_1 = S1
        self.s_2 = S2
        self.B1_fc = B1_fc
        
        ### info
        self.locs = locs
        
        
        


    def forward(self, X, V, I, iden):
        ### Ensure typing
        V = V.to(dtype=torch.float32, device=device)

        ### Fast update rule for B
        # bv = torch.matmul(self.B,V)

        # num = torch.matmul(X, V.T)
        # denom = torch.matmul(bv, V.T)
        # denom = denom + torch.ones(denom.shape, device=device)*EPSILON

        # d = torch.div(num,denom)

        # self.B = torch.mul(B,d)
        # # self.B = torch.clamp(B, min=0)
        # B = torch.round(B)

        ### FAST update rule V
        x_difs = self.locs[:,0].unsqueeze(1) - self.locs[:,0].unsqueeze(0)
        y_difs = ((self.locs[:,1].unsqueeze(1)).to("cpu") - (self.locs[:,1].unsqueeze(0)).to("cpu")).to(device)
        t_dis = torch.abs(x_difs)**2 + torch.abs(y_difs)**2
        z_difs = torch.sum(torch.abs(I.unsqueeze(2)-I.unsqueeze(1)), dim=0)

        A = torch.exp(-((t_dis)/((self.s_1)**2+EPSILON) + (z_difs ** 2)/(self.s_2**2+EPSILON)))
        D = torch.diag(torch.sum(A, axis=1))

        
        Jm = torch.ones((X.shape[1], X.shape[1]), dtype=torch.float32, device=device)
        J = torch.ones((self.B1_fc.out_features, X.shape[1]), dtype=torch.float32, device=device)
        E = torch.ones((len(X[0]), self.B1_fc.out_features), dtype=torch.float32, device=device)*EPSILON
        

        b = self.B1_fc(iden)
        numerator = self.B1_fc(X.T) + torch.mul(self.l_1, torch.matmul(A, V.T)) + self.l_2*torch.matmul(Jm, J.T)
        denominator = torch.abs(self.B1_fc(torch.matmul(V.T, b.T)) + self.l_1*torch.matmul(D, V.T) + self.l_2*torch.matmul(torch.matmul(V.T, J), J.T)) +E

        delta = torch.div(numerator, denominator)

        V = torch.mul(V.T, delta).T

        ### Force sum to 1
        V = V / (sum(V) + torch.ones(len(V[0]), device=device)*EPSILON)
        # print(V)
        
        return V, D, A, J, Jm, b



class SuperNet(nn.Module):
    """
    Class for a Regularized DNMF with varying layers number.
    Input:
        -n_layers = number of layers to construct the Net
    each layer is MU of Frobenius norm
    """

    def __init__(self, n_layers, locs):
        super(SuperNet, self).__init__()

        lambda1 = nn.Parameter(torch.ones(1)*32, requires_grad=True)
        lambda2 = nn.Parameter(torch.ones(1)*18, requires_grad=True)

        sigma1 = nn.Parameter(torch.ones(1)*22, requires_grad=True)
        sigma2 = nn.Parameter(torch.ones(1)*8, requires_grad=True)

        B1_fc = nn.Linear(10431,5)
        self.I = torch.eye(B1_fc.in_features, device=device)

        self.n_layers = n_layers
        self.deep_nmfs = nn.ModuleList(
            [SuperLayer(locs, lambda1, lambda2, sigma1, sigma2, B1_fc) for _ in range(self.n_layers)]
        )


    def forward(self, x, v, i):
    # sequencing the layers and forward pass through the network

        for n, l in enumerate(self.deep_nmfs):
            if n >= 0:
                for param in l.parameters():
                    param.requires_grad = True
            else:
                for param in l.parameters():
                    param.requires_grad = False
            # v,l1,l2,s1,s2 = l(x, v, i)  # Updated V is passed to each layer
            v, d, a, j, jm, b = l(x, v, i, self.I)  # Updated V is passed to each layer
            torch.mps.empty_cache()
        
        # print(1*torch.norm(x-torch.matmul(self.B,v), p="fro")**2)
        # print(self.l1*torch.trace(torch.matmul(torch.matmul(v,(d-a)), v.T)))
        # print(self.l2*torch.norm(torch.matmul(v.T,j)-jm, p="fro")**2)
        return v, b


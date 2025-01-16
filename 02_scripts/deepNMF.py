import torch
import torch.nn as nn
import numpy as np

EPSILON = torch.finfo(torch.float32).eps
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# ================================ supervised Net ======================================
class SuperLayer(nn.Module):

    def __init__(self, locs, L1, L2, L3, S1, S2, B, I, supernet): # W1, W2, W3, B, I
        super(SuperLayer, self).__init__()
        self.l_1 = L1
        self.l_2 = L2
        self.l_3 = L3
        self.s_1 = S1
        self.s_2 = S2
        self.locs = locs
        self.I = I.to(device)
        self.B = B

        
        # self.W1 = supernet.W_1.to(device)
        # self.W2 = supernet.W_2.to(device)
        # self.W3 = supernet.W_3.to(device)
        


        


    def forward(self, X, V, W1, W2, W3):
        stack = torch.stack([W1, W2, W3]).to(device=device)
        W = stack[torch.argmax(self.I.T, dim=1)].squeeze().to(device=device)

        V = V.to(dtype=torch.float32, device=device)


        x_difs = self.locs[:,0].unsqueeze(1) - self.locs[:,0].unsqueeze(0)
        y_difs = ((self.locs[:,1].unsqueeze(1)).to("cpu") - (self.locs[:,1].unsqueeze(0)).to("cpu")).to(device)
        t_dis = torch.abs(x_difs)**2 + torch.abs(y_difs)**2
        z_difs = torch.sum(torch.abs(self.I.unsqueeze(2)-self.I.unsqueeze(1)), dim=0)

        A = torch.exp(-((t_dis)/((self.s_1)**2+EPSILON) + (z_difs ** 2)/(self.s_2**2+EPSILON)))
        D = torch.diag(torch.sum(A, axis=1))

        # Jm = torch.ones((len(np.array(X)[0]), len(np.array(X)[0])), dtype=torch.float32, device=device)
        # J = torch.ones((len(np.array(self.b)[0]), len(np.array(X)[0])), dtype=torch.float32, device=device)
        Jm = torch.ones((X.shape[1], X.shape[1]), dtype=torch.float32, device=device)
        J = torch.ones((self.B.shape[1], X.shape[1]), dtype=torch.float32, device=device)
        # B = self.b.to(device)
        E = torch.ones((len(X[0]), len(self.B[0])), dtype=torch.float32, device=device)*EPSILON
        
        numerator = torch.matmul(X.T, self.B) + torch.mul(self.l_1, torch.matmul(A, V.T)) + self.l_2*torch.matmul(Jm, J.T)
        denominator = torch.matmul(torch.matmul(V.T,self.B.T), self.B) + self.l_1*torch.matmul(D, V.T) + self.l_2*torch.matmul(torch.matmul(V.T, J), J.T) +E +2*self.l_3*torch.matmul(V.T,torch.matmul(W.t(),W))


        delta = torch.div(numerator, denominator)

        V = torch.mul(V.T, delta).T

        threshold = 1e-3  # Define the threshold value
        V[V < threshold] = 0
        V = V/sum(V)
        # print(V)
        
        # return torch.mul(V.T, delta).T, self.l_1, self.l_2, self.s_1, self.s_2
        # return torch.mul(V.T, delta).T, D, A, J, Jm, B
        return V, D, A, J, Jm, W



class SuperNet(nn.Module):
    """
    Class for a Regularized DNMF with varying layers number.
    Input:
        -n_layers = number of layers to construct the Net
    each layer is MU of Frobenius norm
    """

    def __init__(self, n_layers, locs, B, I):
        super(SuperNet, self).__init__()

        lambda1 = nn.Parameter(torch.ones(1, device=device)*30, requires_grad=True)
        lambda2 = nn.Parameter(torch.ones(1, device=device)*50, requires_grad=True)
        lambda3 = nn.Parameter(torch.ones(1, device=device)*30, requires_grad=True)

        sigma1 = nn.Parameter(torch.ones(1, device=device)*5, requires_grad=True)
        sigma2 = nn.Parameter(torch.ones(1, device=device)*5, requires_grad=True)


        self.W_1 = nn.Parameter(torch.ones((1, B.shape[1]), device=device)*20, requires_grad=True)
        self.W_2 = nn.Parameter(torch.ones((1, B.shape[1]), device=device)*8, requires_grad=True)
        self.W_3 = nn.Parameter(torch.ones((1, B.shape[1]), device=device)*.5, requires_grad=True)



        # self.W_3_norm = torch.softmax(self.W_3, dim=-1)

        # self.l1 = lambda1
        # self.l2 = lambda2

        self.n_layers = n_layers
        self.deep_nmfs = nn.ModuleList(
            [SuperLayer(locs, lambda1, lambda2, lambda3, sigma1, sigma2, B, I, self) for _ in range(self.n_layers)] #W_1,W_2,W_3, B, I
        )


    def forward(self, x, v):
    # sequencing the layers and forward pass through the network
        for n, l in enumerate(self.deep_nmfs):
            if n >= 0:
                for param in l.parameters():
                    param.requires_grad = True
            else:
                for param in l.parameters():
                    param.requires_grad = False
            # v,l1,l2,s1,s2 = l(x, v, i)  # Updated V is passed to each layer
            v, d, a, j, jm, w = l(x, v, self.W_1.abs() / self.W_1.abs().sum(), self.W_2.abs() / self.W_2.abs().sum(), self.W_3.abs() / self.W_3.abs().sum())  # Updated V is passed to each layer
        # print(1*torch.norm(x-torch.matmul(self.B,v), p="fro")**2)
        # print(self.l1*torch.trace(torch.matmul(torch.matmul(v,(d-a)), v.T)))
        # print(self.l2*torch.norm(torch.matmul(v.T,j)-jm, p="fro")**2)
        return v, w


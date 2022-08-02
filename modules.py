import pdb
import dgl
import torch 
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from torch.nn import init
import json
 

class Unfolding(nn.Module):


    def __init__(self, alp=0.5, prop_step=5, kernel='DAD', clamp=False):

        super().__init__()

        self.alp = alp 
        self.prop_step = prop_step
        self.kernel = kernel
        self.clamp = clamp
        self.post_step = lambda x:torch.clamp(x, 0, 1)
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



    def DADX(self, graph, X):
        """Y = D^{-1/2}AD^{-1/2}X"""
        Y = self.D_power_X(graph, X, -0.5)  # Y = D^{-1/2}X
        Y = self.AX(graph, Y)  # Y = AD^{-1/2}X
        Y = self.D_power_X(graph, Y, -0.5)  # Y = D^{-1/2}AD^{-1/2}X
        return Y


    def DAX(self, graph, X):
        """Y = D^{-1/2}D^{-1/2}AX"""
        Y = self.AX(graph, X)  # Y = AX
        Y = self.D_power_X(graph, Y, -0.5)  # Y = D^{-1/2}AX
        Y = self.D_power_X(graph, Y, -0.5)  # Y = D^{-1/2}D^{-1/2}AX
        return Y


    def ADX(self, graph, X):
        """Y = AD^{-1/2}D^{-1/2}X"""
        Y = self.D_power_X(graph, X, -0.5)  # Y = D^{-1/2}X
        Y = self.D_power_X(graph, Y, -0.5)  # Y = D^{-1/2}D^{-1/2}X
        Y = self.AX(graph, Y)  # Y = AD^{-1/2}D^{-1/2}X
        return Y



    def AX(self, graph, X):
        """Y = AX"""

        graph.srcdata["h"] = X
        graph.update_all(
            fn.u_mul_e("h", "w", "m"), fn.sum("m", "h"),
        )
        Y = graph.dstdata["h"]

        return Y


    def D_power_X(self, graph, X, power):
        """Y = D^{power}X"""

        degs = graph.ndata["deg"]
        norm = torch.pow(degs, power)
        
        Y = X * norm.view(X.size(0), 1)
        return Y




    def forward(self, g, X, cat=False):
        
        X = X.to(g.device)

        Y = X

        res = []
        res.append(Y)


        g.edata["w"]    = torch.ones(g.number_of_edges(), 1, device = g.device)
        g.ndata["deg"]  = g.in_degrees().float().clamp(min=1)
        for i in range(self.prop_step):

            if self.kernel == 'DAD':
                Y = self.alp * self.DADX(g, Y) + (1 - self.alp) * X
            elif self.kernel == 'AD':
                Y = self.alp * self.ADX(g, Y) + (1 - self.alp) * X
            elif self.kernel == 'DA':
                Y = self.alp * self.DAX(g, Y) + (1 - self.alp) * X


            # if i % 2 == 1:
            #     if cat == True:
            #         res.append(Y)
            
            if cat == True:
                res.append(Y)


        if cat == True:
            return res
        else:
            return Y.cpu()










class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)




import os
import shutil
import psutil 
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_interactions import ioff, panhandler, zoom_factory
import plotly.express as px

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import GCNConv, GATConv, global_add_pool
from scipy.special import expit

from configs import *

"""
Crystal graph convolutional neural network layers and model

General Dimensional Parameters: 
    n       number of nodes in crystal graph
    b       number of bonds in crystal graph
"""

# GRAPH CONVOLUTIONAL LAYERS
class GraphConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        """
        Parameters:
            dim_in       number of input features
            dim_out      number of output features 
        """

        super(GraphConvLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.conv1 = GCNConv(dim_in, dim_out)
        self.double()

    # Forward pass
    def forward(self, x_data, edge_index, bond_dist):
        """
        Parameters:
            x_data               node feature
            edge_index           bond edge indices (corresponds to node indices)
            bond_dist            bond distances

        Return:
            x                    new node features after graph attention layers
        """

        def cumulative_edgeIndex(x_data, edge_index, bond_dist):
            
            S, _, B2 = edge_index.size()            
            N = 558                                 
            
            edge_index_batched = []
            bond_dist_batched = []
            for s in range(S):
                b = sum(edge_index[s, 0, k] != edge_index[s, 1, k] for k in range(B2))  
                edge_index_strip = edge_index[s, :, :b]                 # unpadded
                edge_index_batched.append(edge_index_strip + (s * N))
                
                bond_dist_strip = bond_dist[s, :b]
                bond_dist_batched.append(bond_dist_strip)
                
            edge_index_batched = torch.cat(edge_index_batched, dim=-1)
            bond_dist_batched = torch.cat(bond_dist_batched, dim=0)
                        
            return edge_index_batched, bond_dist_batched

        # one large graph with S disconnected graphs
        edge_indx, bond_distance = cumulative_edgeIndex(x_data, edge_index, bond_dist)
                
        x = x_data.view(-1, x_data.size(-1)).double()    # stacks all features of batched S crystals together (1st 588 nodes = crystal 1, etc.)
        
        edge_indx = edge_indx.long()
        bond_distance = bond_distance.double()                 
        bond_distance = bond_distance/max(bond_distance)                # normalized
                
        x = self.conv1(x, edge_indx, bond_distance)

        return x
    
# GRAPH ATTENTION LAYERS
class GraphAttentionLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        """
        Parameters:
            dim_in       number of input features
            dim_out      number of output features 
        """

        super(GraphAttentionLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.att1 = GATConv(dim_in, dim_out, heads=n_heads, dropout=dropout)
        self.double()

    # Forward pass
    def forward(self, x_data, edge_index, bond_dist):
        """
        Parameters:
            x_data               node feature
            edge_index           bond edge indices (corresponds to node indices)
            bond_dist            bond distances

        Return:
            x                    new node features after graph attention layers
        """

        def cumulative_edgeIndex(x_data, edge_index, bond_dist):
            
            S, _, B2 = edge_index.size()            
            N = 558                                 
            
            edge_index_batched = []
            bond_dist_batched = []
            for s in range(S):
                b = sum(edge_index[s, 0, k] != edge_index[s, 1, k] for k in range(B2))  
                edge_index_strip = edge_index[s, :, :b]                 # unpadded
                edge_index_batched.append(edge_index_strip + (s * N))
                
                bond_dist_strip = bond_dist[s, :b]
                bond_dist_batched.append(bond_dist_strip)
                
            edge_index_batched = torch.cat(edge_index_batched, dim=-1)
            bond_dist_batched = torch.cat(bond_dist_batched, dim=0)
                        
            return edge_index_batched, bond_dist_batched

        # one large graph with S disconnected graphs
        edge_indx, bond_distance = cumulative_edgeIndex(x_data, edge_index, bond_dist)
                
        x = x_data.view(-1, x_data.size(-1)).double()    # stacks all features of batched S crystals together (1st 588 nodes = crystal 1, etc.)
        
        edge_indx = edge_indx.long()
        bond_distance = bond_distance.double()                 
        bond_distance = bond_distance/max(bond_distance)                # normalized
                
        x = self.att1(x, edge_indx, bond_distance)

        return x
    
# DIMENSIONAL PROJECTION W/ MLP
class DimensionalProjection(nn.Module):
    def __init__(self, dim_in, dim_out):
        """
        Parameters:
            dim_in    ()    number of input (nodal + bond) features
            dim_out    ()    number of out features (not sure)
        """

        super(DimensionalProjection, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        self.mlp_project = nn.Linear(self.dim_in, self.dim_out)
        self.double()

    # Forward pass
    def forward(self, x_data):
        x = self.mlp_project(x_data)

        return x    

# CRYSTAL GRAPH CONVOLUTIONAL NEURAL NETWORK (CGCNN) MODEL
class CGCNNModel(nn.Module):
    def __init__(self, structureParams):
        """
        Parameters:
            dim_in           ()      number of input (nodal + bond) features
            dim_out           ()      number of out features (not sure)
            dim_hidFeat         ()      number of hidden features after pooling

            n_convLayer         ()      number of convolutional layers
            n_hidLayer_pool     ()      number of hidden layers after pooling
        """

        super(CGCNNModel, self).__init__()

        dim_pressureFeat = structureParams["dim_pressureFeat"]
        dim_texturalFeat = structureParams["dim_texturalFeat"]

        dim_in = structureParams["dim_in"]
        dim_out = structureParams["dim_out"]
        dim_hidFeat = structureParams["dim_hidFeat"]
        n_convLayer = structureParams["n_convLayer"]
        n_hidLayer_pool = structureParams["n_hidLayer_pool"]
        
        n_attLayer = structureParams["n_attLayer"]
        dim_att = structureParams["dim_att"]
        
        dim_isotherm = structureParams["dim_fc_out"]

        self.dim_pressureFeat = dim_pressureFeat
        self.dim_texturalFeat = dim_texturalFeat

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_att = dim_att
        self.dim_hidFeat = dim_hidFeat
        self.dim_isotherm = dim_isotherm
                

        self.conv = nn.ModuleList([GraphConvLayer(dim_in=(self.dim_att[-1]*n_heads)+self.dim_texturalFeat+self.dim_pressureFeat, dim_out=self.dim_out[0])])
        # self.conv = nn.ModuleList([GraphConvLayer(dim_in=self.dim_in, dim_out=self.dim_out[0])])
        for i in range(1, n_convLayer):
            self.conv.append(GraphConvLayer(dim_in=self.dim_out[i-1], dim_out=self.dim_out[i]))
          
        self.att = nn.ModuleList([GraphAttentionLayer(dim_in=self.dim_in, dim_out=self.dim_att[0])])
        # self.att = nn.ModuleList([GraphAttentionLayer(dim_in=self.dim_out[-1], dim_out=self.dim_att[0])])
        for i in range(1, n_attLayer):
            self.att.append(GraphAttentionLayer(dim_in=self.dim_att[i-1]*n_heads, dim_out=self.dim_att[i]))
            
        self.batchNorm_hid = []
        self.conv_hid = nn.ModuleList([nn.Linear(self.dim_out[-1], self.dim_hidFeat[0])])       
        # self.conv_hid = nn.ModuleList([nn.Linear((self.dim_att[-1]*n_heads), self.dim_hidFeat[0])])        
        for i in range(1, n_hidLayer_pool):
            hidden_layer = nn.Linear(self.dim_hidFeat[i-1], self.dim_hidFeat[i])
            self.conv_hid.append(hidden_layer)
            
            batchNorm_hid = nn.BatchNorm1d(self.dim_hidFeat[i])
            self.batchNorm_hid.append(batchNorm_hid)
                        
        self.fc_out_isotherm = nn.Linear(self.dim_hidFeat[n_hidLayer_pool-1], self.dim_isotherm)        
        

    # Forward pass:
    def forward(self, x_structure, batchAssign):
        """
        Parameters:
            x_structure
            batch

            node_feat       (N, P_N)     nodal features (original)
            bond_feat      ()           node features
            connectivity    (N, N)       connectivity matrix
        Return:
            x               (N, ?)       new node features with aggregated mean
        """
        
        x_node = x_structure[0]             # (S, N, d)
        x_bond = x_structure[1]
        x_textural = x_structure[2]         # (S, k) 
        x_pressure = x_structure[3]
        
        edge_index = x_bond[:, 0:2, :]  
        bond_dist = x_bond[:, 2, :]
        
        x_textural = DimensionalProjection(dim_in=x_textural.size(1), dim_out=x_textural.size(1)*2)(x_textural.double())   # (S, k')
        x_textural = nn.ReLU()(x_textural)
        x_textural = DimensionalProjection(dim_in=x_textural.size(1), dim_out=x_textural.size(1)*3)(x_textural.double())   # (S, k') 
        x_textural = nn.ReLU()(x_textural)
        x_textural = DimensionalProjection(dim_in=x_textural.size(1), dim_out=x_textural.size(1)*4)(x_textural.double())   # (S, k') 
        x_textural = nn.ReLU()(x_textural)
        
        
        x_pressure = DimensionalProjection(dim_in=x_pressure.size(1), dim_out=x_pressure.size(1)*2)(x_pressure.double())
        x_pressure = nn.ReLU()(x_pressure)
        x_pressure = DimensionalProjection(dim_in=x_pressure.size(1), dim_out=x_pressure.size(1)*3)(x_pressure.double())
        x_pressure = nn.ReLU()(x_pressure)

        S, N, _ = x_node.size()
        
        textural_feat_2d = []
        pressure_feat_2d = []
        textural_feat_3d = []
        pressure_feat_3d = []
        for s in range(S):
            textural_repeat_2d = x_textural[s, :].repeat(N, 1)
            textural_feat_2d.append(textural_repeat_2d)
            
            textural_repeat_3d = x_textural[s, :].unsqueeze(0).repeat(N, 1)
            textural_feat_3d.append(textural_repeat_3d)
            
            pressure_repeat_2d = x_pressure[s, :].repeat(N, 1)
            pressure_feat_2d.append(pressure_repeat_2d)
            
            pressure_repeat_3d = x_pressure[s, :].unsqueeze(0).repeat(N, 1)
            pressure_feat_3d.append(pressure_repeat_3d)
            
        textural_feat_2d = torch.cat(textural_feat_2d, dim=0)             # (S, k')
        pressure_feat_2d = torch.cat(pressure_feat_2d, dim=0)
                    
        textural_feat_3d = torch.stack(textural_feat_3d, dim=0)             # (S, N, k')
        pressure_feat_3d = torch.stack(pressure_feat_3d, dim=0)     # need to make sure textural feat not in empty/padded nodes!!!
          
        # x = torch.cat((x_node, textural_feat_3d), dim=-1)    # concatenate node and global textural features (to each node) (S, N, d+k)
        # x = torch.cat((x_node, textural_feat_3d, pressure_feat_3d), dim=-1)    # concatenate node and global textural features (to each node) (S, N, d+k)
                
        x = x_node
                        
        for attLayer in self.att:                         # reuses weights between layers  
            x = attLayer(x, edge_index, bond_dist)
            x = nn.BatchNorm1d(attLayer.dim_out*n_heads)(x)
            x = nn.ReLU()(x)
            
        x = torch.cat((x, textural_feat_2d, pressure_feat_2d), dim=-1) 
            
        for convLayer in self.conv:                         # reuses weights between layers  
            x = convLayer(x, edge_index, bond_dist)
            x = nn.BatchNorm1d(convLayer.dim_out)(x)
            x = nn.ReLU()(x)
            
        x = global_add_pool(x, batchAssign)   

        for k, hidLayer in enumerate(self.conv_hid):             
            x = hidLayer(x)
            x = nn.ReLU()(x)
            
        x_isotherm = self.fc_out_isotherm(x)                
        
        return x_isotherm
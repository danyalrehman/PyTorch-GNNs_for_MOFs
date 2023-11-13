import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_add_pool

"""
Crystal graph convolutional neural network layers and model

General Dimensional Parameters: 
    n       number of nodes in crystal graph
    b       number of bonds in crystal graph
    a       number of bond pairs in crystal graph
    P_N     number of nodal properties/features
    P_B     number of bond properties/features
    P_A     number of angle properties/features
    P_T     number of textural properties
"""


# GRAPH CONVOLUTIONAL LAYERS
class GraphConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        """
        Parameters:
            dim_in    ()    number of input (nodal + bond) features
            dim_out    ()    number of out features (not sure)
        """

        super(GraphConvLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.conv1 = GCNConv(dim_in, dim_out)
        self.double()

    # Forward pass
    def forward(self, x, connectivity):
        """
        Parameters:
            node_feat      ()      node feature
            bond_feat    ()        bond feature
            connectivity  (N, N)    connectivity matrix

        Return:
            x             (same as node_feat)        new node features with aggregated mean
        """
                 
        def connectivity_to_edgeIndex(connectivity_matrix):
            S, N, _ = connectivity_matrix.size()
            edge_index_batched = []

            cumulative_nodes = 0
            for s in range(S):
                src, trg = torch.where(connectivity_matrix[s] != 0)
                src += cumulative_nodes
                trg += cumulative_nodes

                edges = torch.stack([src, trg], dim=0)
                edge_index_batched.append(edges)
                cumulative_nodes += N

            edge_index_batched = torch.cat(edge_index_batched, dim=1)
            return edge_index_batched

        # one large graph with S disconnected graphs
        edge_indx = connectivity_to_edgeIndex(connectivity)
        
        x = x.view(-1, x.size(-1)).double()        # stacks all the features of batched S crystals together (1st 588 nodes = crystal 1, 2nd 588 nodes = crystal 2, etc.)
        x = self.conv1(x, edge_indx)
        # x = torch.relu(self.conv1(x, edge_indx))

        return x
    
    def update_input_dim(self, new_dim):
        self.dim_in = new_dim
        self.conv1.weight = nn.Parameter(torch.Tensor(new_dim, self.dim_out))            # update weight matrix in GCNConv layer


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

        dim_in = structureParams["dim_in"]
        dim_out = structureParams["dim_out"]
        dim_hidFeat = structureParams["dim_hidFeat"]
        n_convLayer = structureParams["n_convLayer"]
        n_hidLayer_pool = structureParams["n_hidLayer_pool"]
        
        dim_isotherm = structureParams["dim_fc_out"][0]
        dim_enthalpy = structureParams["dim_fc_out"][1]
        dim_enthalpy_LB = structureParams["dim_fc_out"][2]
        dim_enthalpy_UB = structureParams["dim_fc_out"][3]

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidFeat = dim_hidFeat

        self.dim_isotherm = dim_isotherm
        self.dim_enthalpy = dim_enthalpy
        self.dim_enthalpy_LB = dim_enthalpy_LB
        self.dim_enthalpy_UB = dim_enthalpy_UB

        self.conv = nn.ModuleList([GraphConvLayer(dim_in=self.dim_in, dim_out=self.dim_out[0])])
        for i in range(1, n_convLayer):
            self.conv.append(GraphConvLayer(dim_in=self.dim_out[i-1], dim_out=self.dim_out[i]))
            
        self.conv_hid = nn.ModuleList([nn.Linear(self.dim_out[n_convLayer-1], self.dim_hidFeat[0])])
        for i in range(1, n_hidLayer_pool):
            self.conv_hid.append(nn.Linear(self.dim_hidFeat[i-1], self.dim_hidFeat[i]))
            
        self.fc_out_isotherm = nn.Linear(self.dim_hidFeat[n_hidLayer_pool-1], self.dim_isotherm)
        self.fc_out_enthalpy = nn.Linear(self.dim_hidFeat[n_hidLayer_pool-1], self.dim_enthalpy)
        self.fc_out_enthalpy_LB = nn.Linear(self.dim_hidFeat[n_hidLayer_pool-1], self.dim_enthalpy_LB)
        self.fc_out_enthalpy_UB = nn.Linear(self.dim_hidFeat[n_hidLayer_pool-1], self.dim_enthalpy_UB)
        
        
        
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

        node_feat, bond_feat, connectivity = x_structure[0], x_structure[1], x_structure[2]
        bond_feat = bond_feat[:, :node_feat.size(1), :node_feat.size(1)]

        x = torch.cat((node_feat, bond_feat), dim=-1)   
        
        i = 0
        for convLayer in self.conv:                         # reuses weights between layers            
            convLayer.update_input_dim(self.dim_out[i])
            x = convLayer(x, connectivity)
            self.dim_in = convLayer.dim_out
            i += 1
            
        x = global_add_pool(x, batchAssign)
        x = nn.Softplus()(x)

        # is this where you concatenate textural to learned crytsal features x?

        i = 0
        for hidLayer in self.conv_hid:            
            convLayer.update_input_dim(self.dim_hidFeat[i])
            x = hidLayer(x)
            i += 1

        x_isotherm = self.fc_out_isotherm(x)
        x_enthalpy = self.fc_out_enthalpy(x)
        x_enthalpy_LB = self.fc_out_enthalpy_LB(x)
        x_enthalpy_UB = self.fc_out_enthalpy_UB(x)
        
        return x_isotherm, x_enthalpy, x_enthalpy_LB, x_enthalpy_UB
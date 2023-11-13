import numpy as np
import pandas as pd

from rdkit import Chem
import networkx as nx

import torch
import torch.nn.functional as F
from torch import Tensor

"""
Data pre-processing and reshaping 

Input Features: textural properties, nodal info., bond info., angle, connectivity matrix
    N       maximum number of nodes (across all crystal structures)
    B       maximum number of bonds (across all crystal structures)
    A       maximum number of bond pairs (across all crystal structures)
    P_N     number of nodal properties/features
    P_B     number of bond properties/features
    P_A     number of angle properties/features
    P_T     number of textural properties
    S       number of crystal structures 

Outputs: isotherm fitting parameters, enthalpy (& errors) fitting parameters
    n_iso   number of isotherm fitting parameters
    n_H     number of enthalpy fitting parameters 
"""

dataDir = "C:/Users/Emily/ML-Adsorption/"

def mol_nx(mol):
    nxGraph = nx.Graph()
    for atom in mol.GetAtoms():
        nxGraph.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum(), is_aromatic=atom.GetIsAromatic(),
                         atom_symbol=atom.GetSymbol())
    for bond in mol.GetBonds():
        nxGraph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondType(),
                         in_ring=bond.IsInRing())
    return nxGraph


# INPUTS
def structure_inputs(dataDir):
    def bondDist(x1, x2, y1, y2, z1, z2):
        dist = np.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2) + ((z2 - z1) ** 2))
        return dist

    texturalProp_data = pd.read_excel(dataDir + "texturalProperties.xlsx")

    uniqueElements = []
    for s in texturalProp_data.iloc[:, 0]:
        uniqueElements = uniqueElements + list(list(
            (pd.read_csv(dataDir + "mol/" + str(s) + ".mol", sep="\s+", skiprows=[0, 1, 2]).iloc[:, 0:4].dropna()).iloc[:,
            3].to_numpy()))
    uniqueElements = np.unique(uniqueElements)

    N = max([mol_nx(Chem.MolFromMolFile(dataDir + "mol/" + str(s) + ".mol", sanitize=False)).number_of_nodes() for s in
             texturalProp_data.iloc[:, 0]])  # Max. number of nodes across all crystal structures

    nodeFeat = []
    bondFeat = []
    connectivityFeat = []

    for s in texturalProp_data.iloc[:, 0]:
        m = Chem.MolFromMolFile(dataDir + "mol/" + str(s) + ".mol", sanitize=False)
        G = mol_nx(m)
        n = G.number_of_nodes()

        xyz = ((pd.read_csv(dataDir + "mol/" + str(s) + ".mol", sep="\s+", skiprows=[0, 1, 2])).iloc[:, 0:4].dropna()).iloc[
              :, 0:3].to_numpy()
        elements = ((pd.read_csv(dataDir + "mol/" + str(s) + ".mol", sep="\s+", skiprows=[0, 1, 2])).iloc[:,
                    0:4].dropna()).iloc[:, 3].to_numpy()

        # Node Features: Elements [one-hot encoding], shape: (n, len(uniqueElements))  |   Rings [one-hot encoding], shape: (n, )
        nodeOHE = torch.zeros((n, len(uniqueElements)))
        nodeRingOHE = torch.zeros(n)

        for i in range(n):
            indx = list(uniqueElements).index(elements[i])
            nodeOHE[i, indx] = 1

        nodePairs_tot = []
        ring = nx.get_edge_attributes(G, "in_ring")
        for i in range(len(ring.values())):
            if list(ring.values())[i]:
                nodePairs_tot = nodePairs_tot + str(list(ring.keys())[i]).replace("(", "").replace(")", "").replace(" ",
                                                                                                                    "").split(
                    ",")
        indx = [int(x) for x in np.unique(nodePairs_tot)]
        nodeRingOHE[indx] = 1
        nodeRingOHE = nodeRingOHE.view(-1, 1)                 # convert from 1D to 2D tensor

        nodeFeat_cat = torch.cat((nodeOHE, nodeRingOHE), dim=1)  # concatenate element and ring features
        nodeFeat_pad = F.pad(input=nodeFeat_cat, pad=(0, 0, 0, N - n), mode="constant",
                             value=0)  # Pad with zeros: shape (N, len(uniqueElements) + 1)
        nodeFeat.append(nodeFeat_pad)

        # Bond Features: Bond distances, shape: (n, n)
        distMat = torch.zeros((n, n))
        for i in range(n):
            xi = float(xyz[i][0])
            yi = float(xyz[i][1])
            zi = float(xyz[i][2])
            for j in range(n):
                xj = float(xyz[j][0])
                yj = float(xyz[j][1])
                zj = float(xyz[j][2])
                distMat[i, j] = bondDist(xi, xj, yi, yj, zi, zj)

        bondFeat_pad = F.pad(input=distMat, pad=(0, N - n, 0, N - n), mode="constant",
                             value=0)  # Pad with zeros: shape (N, N)
        bondFeat.append(bondFeat_pad)

        # Connectivity/Adjacency matrix: shape (n, n)
        connectivity = Tensor(nx.to_numpy_array(G))
        connectivity_pad = F.pad(input=connectivity, pad=(0, N - n, 0, N - n), mode="constant",
                                 value=0)  # Pad with zeros: shape (N, N)
        connectivityFeat.append(connectivity_pad)

    nodeFeat = torch.stack(nodeFeat, dim=0)  # shape: (S, N, P_N)
    bondFeat = torch.stack(bondFeat, dim=0)  # shape: (S, N, N)
    connectivityFeat = torch.stack(connectivityFeat, dim=0)  # shape: (S, N, N)

    # Save all data
    data_dict = {
        "nodeFeat": nodeFeat,
        "bondFeat": bondFeat,
        "connectivityFeat": connectivityFeat
    }
    torch.save(data_dict, dataDir + "X_dataset.pth")

    # Load and access all data
    loadData = torch.load(dataDir + "X_dataset.pth")

    # Access the tensors and variables
    nodeFeat = loadData["nodeFeat"]
    bondFeat = loadData["bondFeat"]
    connectivityFeat = loadData["connectivityFeat"]

    return N, loadData

# OUTPUTS
def structure_outputs(dataDir):
    # OUTPUTS
    y_data = pd.read_excel(dataDir + "outputData.xlsx")  # shape: (S, n_iso+(3*n_H))
    qm, K, n = y_data.iloc[:, 1], y_data.iloc[:, 2], y_data.iloc[:, 3]
    y1_H, y2_H, y3_H = y_data.iloc[:, 4], y_data.iloc[:, 5], y_data.iloc[:, 6]
    y1_H_LB, y2_H_LB, y3_H_LB = y_data.iloc[:, 7], y_data.iloc[:, 8], y_data.iloc[:, 9]
    y1_H_UB, y2_H_UB, y3_H_UB = y_data.iloc[:, 10], y_data.iloc[:, 11], y_data.iloc[:, 12]
    
    y_data = torch.tensor(y_data.iloc[:, 1:].values, dtype=torch.float32)

    return y_data

def get_params():
    dataDir = "/pool001/elin49/qmof_mol/"
    texturalProp_data = pd.read_excel("/home/elin49/CGCNN/texturalProperties.xlsx")
    N = max([mol_nx(Chem.MolFromMolFile(dataDir + str(s) + ".mol", sanitize=False)).number_of_nodes() for s in
             texturalProp_data.iloc[:, 0]])  # Max. number of nodes across all crystal structures
    return N
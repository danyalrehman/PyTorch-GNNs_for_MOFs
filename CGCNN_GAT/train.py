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
import CGCNN_GAT 


"""
Training, testing, and validation of CGCNN model
"""

# Get system memory information
mem = psutil.virtual_memory()
print(f"Total Memory: {mem.total / (1024 ** 3):.2f} GB", flush=True)
print(f"Available Memory: {mem.available / (1024 ** 3):.2f} GB", flush=True)

class TrainCGCNN():
    """
    Train crystal graph convolutional neural network (CGCNN)
    """

    def __init__(self):      
        # Initialize general parameters
        job_path = os.path.join(jobPath, jobName)         # log directory
        if not os.path.exists(jobPath):
            os.makedirs(jobPath)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")          # device
        print(self.device, flush=True)
        
        self.dtype = torch.set_default_dtype(torch.float64 if fp64 else torch.float32)        # dtype

        if train:
            if run_dataProcess:
                _, structureInputs = dataProcess_isotherm.structure_inputs(dataDir=dataPath)
            else:
                print("Loading structural data...", flush=True)
                structureInputs = torch.load(dataPath+"X_dataset_electro.pth")
                
            print("Building train, validation, and test data...", flush=True)
            nodeFeat, bondFeat, pressureFeat = structureInputs["nodeFeat"][:,:,:-19], structureInputs["bondFeat"], structureInputs["nodeFeat"][:,:,-19:]
            bondFeat = bondFeat[:, :nodeFeat.size(1), :nodeFeat.size(1)]
            
            texturalProp_data = pd.read_excel(jobPath + "texturalProperties_vol.xlsx")
            texturalFeat = torch.tensor(texturalProp_data.iloc[:,1:].values, dtype=torch.float32)
            
            pressureFeat = pressureFeat[:, 0, :]            
            pressureFeat = torch.index_select(pressureFeat, dim=1, index=pos)
            
            H_data = torch.load("H_dataset.pth")["H"]
            H_data = torch.index_select(H_data, dim=1, index=pos)
            
            loadData = torch.load(jobPath + "y_dataset19.pth")
            y_data = loadData["isotherm"]
            y_data = torch.index_select(y_data, dim=1, index=pos)
            
            if len(num) == 1:
                self.train_batchSize = 1
                self.val_batchSize = train_batchSize
                self.test_batchSize = train_batchSize

                nodeFeat = nodeFeat[num[0], :, :].unsqueeze(0)
                bondFeat = bondFeat[num[0], :, :].unsqueeze(0)
                pressureFeat = pressureFeat[num[0], :, :].unsqueeze(0)
                texturalFeat = texturalFeat[num[0], :].unsqueeze(0)
                H_data = H_data[num[0], :].unsqueeze(0)
                y_data = y_data[num[0], :].unsqueeze(0)

                y_train_isotherm = y_data.double()
                y_val_isotherm = y_data.double()  
                y_test_isotherm = y_data.double()  

                train_dataset = TensorDataset(nodeFeat, bondFeat, texturalFeat, pressureFeat, y_train_isotherm)
                val_dataset = TensorDataset(nodeFeat, bondFeat, texturalFeat, pressureFeat, y_val_isotherm)
                self.train_DataLoader = DataLoader(train_dataset, batch_size=self.train_batchSize, shuffle=True, pin_memory=True)
                self.val_DataLoader = DataLoader(val_dataset, batch_size=self.val_batchSize, shuffle=True, pin_memory=True)
            
            else:
                nodeFeat = nodeFeat[num[0]:num[1], :, :]
                bondFeat = bondFeat[num[0]:num[1], :, :]
                y_data = y_data[num[0]:num[1], :]
                
                texturalFeat = texturalFeat[num[0]:num[1], :]
                pressureFeat = pressureFeat[num[0]:num[1], :]
                
                H_data = H_data[num[0]:num[1], :]
                
                # pressureFeat = torch.cat((pressureFeat, H_data), dim=1)            # thermodnyamic properties
                                                                
                x_node_train, x_node_val, x_bond_train, x_bond_val, x_textural_train, x_textural_val, x_pressure_train, x_pressure_val, y_train, y_val = train_test_split(nodeFeat, bondFeat, texturalFeat, pressureFeat, y_data, test_size=0.2, random_state=42)
                x_node_train, x_node_test, x_bond_train, x_bond_test, x_textural_train, x_textural_test, x_pressure_train, x_pressure_test, y_train, y_test = train_test_split(x_node_train, x_bond_train, x_textural_train, x_pressure_train, y_train, test_size=0.2, random_state=42)
            
                train_dataset = TensorDataset(x_node_train, x_bond_train, x_textural_train, x_pressure_train, y_train)
                val_dataset = TensorDataset(x_node_val, x_bond_val, x_textural_val, x_pressure_val, y_val)
                self.train_DataLoader = DataLoader(train_dataset, batch_size=train_batchSize, shuffle=True, pin_memory=True)
                self.val_DataLoader = DataLoader(val_dataset, batch_size=val_batchSize, shuffle=True, pin_memory=True)

        if test:
            if len(num) == 1:
                test_dataset = TensorDataset(nodeFeat, bondFeat, texturalFeat, pressureFeat, y_test_isotherm)
                self.test_DataLoader = DataLoader(test_dataset, batch_size=self.test_batchSize, shuffle=False, pin_memory=True)
            else:
                test_dataset = TensorDataset(x_node_test, x_bond_test, x_textural_test, x_pressure_test, y_test)
                self.test_DataLoader = DataLoader(test_dataset, batch_size=test_batchSize, shuffle=False, pin_memory=True)
            
        print("Done building train, validation, and test data.", flush=True)

        # Initialize model
        structureParams = {
            "dim_texturalFeat": texturalFeat.size(1)*2*3*4, 
            "dim_pressureFeat": pressureFeat.size(1)*2*3, 
            
            "dim_in": nodeFeat.size(2),    # number of features you input (node + TEXTURAL FEATURES)
            
            "n_convLayer": 2,
            "dim_out": [256, 128],   # 1024, 512, 256
            
            "n_attLayer": 2,
            "dim_att": [512, 256],          # 128, 64

            "n_hidLayer_pool": 7,
            "dim_hidFeat": [128, 64, 32, 16, 32, 64, 128],
            
            "dim_fc_out": y_data.size(1)
        }

        self.model = CGCNNModel(structureParams)
        self.model.to(self.device)
        
        self.N = nodeFeat.size(1)            # max number of nodes across all crystal structures

        # Initialize optimizer and scheduler
        if optimizer in ["sgd", "SGD"]:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        elif optimizer in ["Adam", "adam"]:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer in ['Adamax', 'adamax']:
            self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=scheduler_gamma)

        # Load checkpoint
        self.logPath = os.path.join(jobPath, "train_log.txt")
        self.start_epoch = 0
        if not disable_checkpt:
            self.statePath = os.path.join(jobPath, "state_dicts")
            if os.path.exists(self.statePath):
                shutil.rmtree(self.statePath)
            if os.path.exists(self.statePath) and len(os.listdir(self.statePath)) > 0:
                for i in range(num_epoch, 0, -1):
                    fileName = os.path.join(self.statePath, f"epoch_{i}_sd.pt")
                    if os.path.isfile(fileName):
                        checkpoint = torch.load(fileName)
                        self.model.load_state_dict(checkpoint["model_state_dict"])
                        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                        self.start_epoch = checkpoint["epoch"]
                        self.model.eval()
                        break
            elif not os.path.exists(self.statePath):
                os.mkdir(self.statePath)

    def calcLoss(self, y_pred_isotherm, y_target_isotherm):
        y_pred_isotherm = y_pred_isotherm.double()
        y_target_isotherm = y_target_isotherm.double()
        
        mse = nn.MSELoss()(y_pred_isotherm, y_target_isotherm)     
        mae = nn.L1Loss()(y_pred_isotherm, y_target_isotherm)   
        huber = nn.SmoothL1Loss()(y_pred_isotherm, y_target_isotherm)   

        return mse, mae, huber

    
    def train(self):        
        N = self.N                      # max number of nodes across all crystal structures
        best_mse = 1e10
        best_mae = 1e10
        
        train_mse_mean = []
        val_mse_mean = []
        train_mae_mean = []
        val_mae_mean = []
        
        iter_stop = 0
        
        epoch_vec = []
        for epoch in tqdm(range(self.start_epoch, num_epoch)):
            if iter_stop < 50:
                iter_stop += 1
                epoch_vec.append(epoch)

                # Train
                train_mse = []
                train_mae = []            
                self.model.train()            
                for batch, (x_node_train, x_bond_train, x_textural_train, x_pressure_train, y_data_isotherm) in (enumerate(self.train_DataLoader)):                
                    batch_size_train = x_node_train.size(0)  # batch size (in training loop - number of crystal structures in the batch)
                    x_data_train = [x_node_train, x_bond_train, x_textural_train, x_pressure_train]

                    batchAssign_train = torch.tensor([b for b in range(batch_size_train) for n in range(N)])

                    y_pred_isotherm = self.model(x_data_train, batchAssign_train).squeeze()
                    mse, mae, huber = self.calcLoss(y_pred_isotherm, y_data_isotherm)

                    self.optimizer.zero_grad()
                    huber.backward()
                    self.optimizer.step()

                    train_mse.append(mse.item())
                    train_mae.append(mae.item())

                train_mse_mean.append(np.mean(train_mse))
                train_mae_mean.append(np.mean(train_mae))

                self.scheduler.step()

                # Validation
                val_mse = []
                val_mae = []
                self.model.eval()
                for batch, (x_node_val, x_bond_val, x_textural_val, x_pressure_val, y_data_isotherm) in enumerate(self.val_DataLoader):
                    batch_size_val = x_node_val.size(0)  # batch size (in training loop - number of crystal structures in the batch)
                    x_data_val = [x_node_val, x_bond_val, x_textural_val, x_pressure_val]

                    batchAssign_val = torch.tensor([b for b in range(batch_size_val) for n in range(N)])
                    y_pred_isotherm = self.model(x_data_val, batchAssign_val).squeeze()
                    mse, mae, huber = self.calcLoss(y_pred_isotherm, y_data_isotherm)

                    val_mse.append(mse.item())
                    val_mae.append(mae.item())

                val_mse_mean.append(np.mean(val_mse))
                val_mae_mean.append(np.mean(val_mae))

                torch.save({
                        "epoch": epoch + 1,
                        "train_mse_mean": train_mse_mean,
                        "train_mae_mean": train_mae_mean,
                        "val_mse_mean": val_mse_mean,
                        "val_mae_mean": val_mae_mean
                    }, os.path.join(jobPath, "losses"+indx_str+".pth"))

                if (np.mean(val_mae) < best_mae) and (np.mean(val_mse) < best_mse):  
                    best_mae = np.mean(val_mae)
                    best_mse = np.mean(val_mse)

                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict()
                    }, os.path.join(jobPath, "best_model"+indx_str+".pth"))
                    
                    iter_stop = 0
                    
            else:
                print("Model exceeded 50 iterations without improvement. BREAK.")
                break
                                    
        plt.figure(figsize=(9,6))
        # plt.plot(epoch_vec, np.log(np.array(train_mse_mean)), alpha=0.4, color="blue", label="Training - MSE")
        # plt.plot(epoch_vec, np.log(np.array(val_mse_mean)), alpha=0.4, color="red", label="Validation - MSE")
        plt.plot(epoch_vec, np.log(np.array(train_mae_mean)), color="blue", label="Training - MAE")
        plt.plot(epoch_vec, np.log(np.array(val_mae_mean)), color="red", label="Validation - MAE")
        plt.xlabel("Epoch")
        plt.ylabel("Log Loss")
        plt.legend()
                
    def load_best_model(self):
        best_model_path = os.path.join(jobPath, "best_model"+indx_str+".pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.start_epoch = checkpoint["epoch"]
            print(f"Loaded best model from epoch {self.start_epoch}", flush=True)
        else:
            print("Best model checkpoint not found.", flush=True)

    def test(self):
        plt.figure(figsize=(9,6))
        n_plot = 0
        
        N = self.N  # max number of nodes across all crystal structures

        N_lines = len(self.test_DataLoader.dataset)
        colormap = cm.get_cmap("rainbow")
        count = 0
                
        # Testing / Validation
        val_mse = []
        val_mae = []
        self.model.eval()
        for batch, (x_node_val, x_bond_val, x_textural_val, x_pressure_val, y_data_isotherm) in enumerate(self.test_DataLoader):
            batch_size_val = x_node_val.size(0)  # batch size (in training loop - number of crystal structures in the batch)
            x_data_val = [x_node_val, x_bond_val, x_textural_val, x_pressure_val]

            batchAssign_val = torch.tensor([b for b in range(batch_size_val) for n in range(N)])        # needs to be repeated batch_size times (0,0,0, ..., batch_size-1, batch_size-1, batch_size-1)

            y_pred_isotherm = self.model(x_data_val, batchAssign_val).squeeze()
            mse, mae, huber = self.calcLoss(y_pred_isotherm, y_data_isotherm)
            val_mse.append(mse.item())
            val_mae.append(mae.item())
            
            print(f"Average MSE {np.mean(val_mse):.4f}, Average MAE: {np.mean(val_mae):.4f}", flush=True)
                        
            P = np.array([1e3,5e3,1e4,5e4,1e5,2e5,3e5,4e5,5e5,7e5,1e6,1.5e6,2e6,2.5e6,3e6,3.5e6,4e6,4.5e6,5e6])*0.00001
            
            
            err_tot = (np.array(y_data_isotherm)) - (y_pred_isotherm.detach().numpy())
            err_LB_tot = (np.array(y_data_isotherm)) - (y_pred_isotherm.detach().numpy())
            err_UB_tot = (np.array(y_data_isotherm)) - (y_pred_isotherm.detach().numpy())
            
            for i in range(y_data_isotherm.size(0)):
                for j, err_j in enumerate(err_tot[i,:]):
                    if err_j > 0:
                        err_UB_tot[i,j] = 0
                    else:
                        err_LB_tot[i,j] = 0
                                                            
            for i in range(y_data_isotherm.size(0)):
                if n_plot > 3:
                    plt.figure(figsize=(9,6))
                    n_plot = 0
                                
                err = (np.array(y_data_isotherm)[i,:]) - (y_pred_isotherm.detach().numpy()[i,:])
                err_LB = (np.array(y_data_isotherm)[i,:]) - (y_pred_isotherm.detach().numpy()[i,:])
                err_UB = (np.array(y_data_isotherm)[i,:]) - (y_pred_isotherm.detach().numpy()[i,:])

                for j, err_j in enumerate(err):
                    if err_j > 0:
                        err_UB[j] = 0
                    else:
                        err_LB[j] = 0
                        
                errors = [np.array(err_LB), -1*np.array(err_UB)]
                                
                color = colormap((count + i) / N_lines)
                                
                plt.errorbar(P[pos], np.array(y_data_isotherm)[i,:], yerr=np.array(errors), linestyle="solid", fmt='o', ecolor=None, capsize=5, lolims=False)
                plt.xlabel("Pressure [bar]")
                plt.ylabel("Uptake [g/g]")
                
                n_plot += 1
                
            count += i
            
            if batch == 0:
                y_pred_isotherm_tot = torch.empty(1, y_pred_isotherm.size(1))
                y_data_isotherm_tot = torch.empty(1, y_data_isotherm.size(1))
                err_LB_arr = torch.empty(1, y_data_isotherm.size(1))
                err_UB_arr = torch.empty(1, y_data_isotherm.size(1))
                
                y_pred_isotherm_tot = torch.cat((y_pred_isotherm_tot, y_pred_isotherm), dim=0)[1:]
                y_data_isotherm_tot = torch.cat((y_data_isotherm_tot, y_data_isotherm), dim=0)[1:]
                err_LB_arr =  torch.cat((err_LB_arr, err_LB_arr), dim=0)[1:]
                err_UB_arr =  torch.cat((err_UB_arr, err_UB_arr), dim=0)[1:]
                                
            else:
                y_pred_isotherm_tot = torch.cat((y_pred_isotherm_tot, y_pred_isotherm), dim=0)
                y_data_isotherm_tot = torch.cat((y_data_isotherm_tot, y_data_isotherm), dim=0)
                err_LB_arr =  torch.cat((err_LB_arr, err_LB_arr), dim=0)
                err_UB_arr =  torch.cat((err_UB_arr, err_UB_arr), dim=0)
            

        torch.save({
            "y_pred_isotherm": y_pred_isotherm_tot,
            "y_data_isotherm": y_data_isotherm_tot,
            "err_LB": err_LB_arr,
            "err_UB": err_UB_arr
        }, os.path.join(jobPath, "results"+indx_str+".pth"))

        # Plot target vs prediction
        plt.figure(figsize=(23,6))
        for i, p in enumerate(P[pos]):
            min_val = min(min(y_pred_isotherm_tot[:, i].detach().numpy()), min(y_data_isotherm_tot[:, i].detach().numpy()))
            max_val = max(max(y_pred_isotherm_tot[:, i].detach().numpy()), max(y_data_isotherm_tot[:, i].detach().numpy()))
            
            plt.subplot(1, 3, i+1)
            plt.axis("equal")
            plt.plot(np.linspace(min_val, max_val), np.linspace(min_val, max_val), linestyle="dashed", color="black")
            plt.scatter(y_pred_isotherm_tot[:, i].detach().numpy(), y_data_isotherm_tot[:, i].detach().numpy(), alpha=0.5, color="green")
            plt.title(f"{p:.2f} bar(s)")


if __name__ == "__main__":    
    trainer = TrainCGCNN()
    trainer.train()
    trainer.load_best_model()
    trainer.test()
    
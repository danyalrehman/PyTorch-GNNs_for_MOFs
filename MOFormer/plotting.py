import torch
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.interpolate import splrep, BSpline
from scipy.interpolate import UnivariateSpline

plt.rcParams["mathtext.fontset"] = "cm"
font = {'size' : 10}; matplotlib.rc('font', **font)
matplotlib.rcParams['font.family'] = ['Gulliver-Regular', 'sans-serif']

import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def monoExp(x, m, t, b):
    return m * np.log(t * x + b)

p0 = (500, 2.,  100)
params, cv = optimize.curve_fit(monoExp, pressure / 1E5, GCMC_failed, p0)
m, t, b = params

plt.figure(figsize=(5.5,4.5))
plt.plot(smooth_pressure, monoExp(smooth_pressure, m, t, b), linewidth=2, color='navy')
plt.plot(pressure / 1E5, GCMC_failed, marker="o", linewidth=0, markersize=6, fillstyle='full', markerfacecolor="white",  markeredgewidth=1.5, color=cmap_range[0], label=r'Adsorption Isotherms', clip_on=False)
plt.title('QMOF-0A7A2FD', fontsize=15)
plt.xlabel(r'Pressure [bar]', fontsize=13); plt.ylabel(r'CO$_2$ Uptake [g/kg]', fontsize=13)
plt.legend(frameon=False, loc='lower right', fontsize=13)
plt.ylim([0,170]); plt.xlim([0,50])
plt.tick_params(direction='in', length=6, top=True, right=True)
plt.tight_layout()
plt.savefig(r'isotherm.pdf')

df = pd.read_excel('isotherms.xlsx')
cmap_range = sns.color_palette('viridis', n_colors=3)

# Low quality predictions
GCMC_failed, GNN_failed = df['GCMC (Failed)'], df['GNN (Failed)']
MAT_PT_failed, MAT_failed = df['MAT pre-trained (Failed)'], df['MAT (Failed)']

# High quality predictions
GCMC_successful, GNN_successful = df['GCMC (Successful)'], df['GNN (Successful)']
MAT_PT_successful, MAT_successful = df['MAT pre-trained (Successful)'], df['MAT (Successful)']

pressure = np.array([1e3,5e3,1e4,5e4,1e5,2e5,3e5,4e5,5e5,7e5,1e6,1.5e6,2e6,2.5e6,3e6,3.5e6,4e6,4.5e6,5e6])
smooth_pressure = np.arange(0, np.max(pressure), 1000) / 1E5

fig = plt.figure(figsize=(10., 3.75))
plt.subplot(131)
plt.axis('off')

plt.subplot(132)
plt.plot([0,180], [0,180], color='black', linewidth=1, linestyle='--')
plt.plot(GCMC_failed, MAT_failed, marker="o", linewidth=0, markersize=6, fillstyle='full', markerfacecolor="white",  markeredgewidth=1.5, color=cmap_range[2], label=r'MAT', clip_on=False)
plt.plot(GCMC_failed, GNN_failed, marker="^", linewidth=0, markersize=6, fillstyle='full', markerfacecolor="white",  markeredgewidth=1.5, color=cmap_range[0], label=r'CGCNN + GAT', clip_on=False)
plt.plot(GCMC_failed, MAT_PT_failed, marker="h", linewidth=0, markersize=6, fillstyle='full', markerfacecolor="white",  markeredgewidth=1.5, color=cmap_range[1], label=r'MAT (MOFormer PT)', clip_on=False)
plt.title('QMOF-0A7A2FD')
plt.xlabel(r'CO$_2$ Uptake (GCMC) [g/kg]'); plt.ylabel(r'CO$_2$ Uptake (Neural Model) [g/kg]')
plt.legend(frameon=False, loc='upper left', fontsize=8)
plt.xlim([0,175]); plt.ylim([0,175])
plt.tick_params(direction='in', length=6, top=True, right=True)

plt.subplot(133)
plt.plot([0,1300], [0,1300], color='black', linewidth=1, linestyle='--')
plt.plot(GCMC_successful, MAT_successful, marker="o", linewidth=0, markersize=6, fillstyle='full', markerfacecolor="white",  markeredgewidth=1.5, color=cmap_range[2], label=r'MAT', clip_on=False)
plt.plot(GCMC_successful, GNN_successful, marker="^", linewidth=0, markersize=6, fillstyle='full', markerfacecolor="white",  markeredgewidth=1.5, color=cmap_range[0], label=r'CGCNN + GAT', clip_on=False)
plt.plot(GCMC_successful, MAT_PT_successful, marker="h", linewidth=0, markersize=6, fillstyle='full', markerfacecolor="white",  markeredgewidth=1.5, color=cmap_range[1], label=r'MAT (MOFormer PT)', clip_on=False)
plt.title('QMOF-C6E12DD')
plt.xlabel(r'CO$_2$ Uptake (GCMC) [g/kg]'); plt.ylabel(r'CO$_2$ Uptake (Neural Model) [g/kg]')
plt.legend(frameon=False, loc='upper left', fontsize=8)
plt.xlim([0,1300]); plt.ylim([0,1300])
plt.tick_params(direction='in', length=6, top=True, right=True)

plt.tight_layout()
plt.savefig(r'Parity_Figures.pdf')
plt.show()
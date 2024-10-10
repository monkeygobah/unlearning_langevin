import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

gamma_vals = [0, 0.0001, 0.1]  
gamma_vals_str = ['0', '1e-4', '1e-1', '1']


f_acc_data = np.array([
    [0.089, 0.115, 0.094], 
    [0.490, 0.516, 0.482], 
    [0.001, 0.047, 0.062], 
    [0.001, 0.047, 0.062], 
    [0.521, 0.549, 0.509], 
    [0.001, 0.047, 0.062], 
    [0.001, 0.047, 0.062]
])

r_acc_data = np.array([
    [0.837, 0.814, 0.810], 
    [0.872, 0.849, 0.855], 
    [0.183, 0.167, 0.174], 
    [0.184, 0.168, 0.174], 
    [0.890, 0.884, 0.888], 
    [0.183, 0.167, 0.174], 
    [0.183, 0.167, 0.174]
])
conditions = [
    'Baseline', 'R', 'F', 'R+F', 
    'R+F+Preprocess', 'R+F-Iter 80', 'R+F-Iter 95'
]

fig, ax = plt.subplots(2, 1, figsize=(6, 8))

sns.heatmap(f_acc_data, annot=True, cmap='Blues_r', xticklabels=gamma_vals, yticklabels=conditions, ax=ax[0], cbar_kws={'orientation': 'horizontal'})
ax[0].set_title(r'F Accuracy ($\phi=1e-2)$')
ax[0].set_xlabel(r'$\gamma$')

sns.heatmap(r_acc_data, annot=True, cmap='Greens', xticklabels=gamma_vals, yticklabels=conditions, ax=ax[1], cbar_kws={'orientation': 'horizontal'})
ax[1].set_title(r'R Accuracy ($\phi=1e-2)$')
ax[1].set_xlabel(r'$\gamma$')

plt.tight_layout()

plt.savefig('ablation_experiment_phi_1e02_heatmap_vert.png', dpi=300)

plt.show()

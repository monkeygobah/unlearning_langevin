import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

gamma_vals = [0, 0.0001, 0.1]  

gamma_vals_str = ['0', '1e-4', '1e-1', '1']

f_acc_data_phi_1e_4 = np.array([
    [0.089, 0.115, 0.094], 
    [0.488, 0.516, 0.481], 
    [0.001, 0.047, 0.062], 
    [0.001, 0.047, 0.062], 
    [0.444, 0.486, 0.460], 
    [0.001, 0.047, 0.062], 
    [0.001, 0.047, 0.062]
])

r_acc_data_phi_1e_4 = np.array([
    [0.837, 0.814, 0.810], 
    [0.872, 0.850, 0.856], 
    [0.183, 0.167, 0.174], 
    [0.184, 0.168, 0.174], 
    [0.877, 0.871, 0.880], 
    [0.183, 0.167, 0.174], 
    [0.183, 0.167, 0.174]
])

conditions = [
    'Baseline', 'R', 'F', 'R+F', 
    'R+F+Preprocess', 'R+F-Iter 80', 'R+F-Iter 95'
]

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

sns.heatmap(f_acc_data_phi_1e_4, annot=True, cmap='Blues_r', xticklabels=gamma_vals_str, yticklabels=conditions, ax=ax[0],cbar_kws={'orientation': 'horizontal'})
ax[0].set_title(r'F Accuracy ($\phi=1e-04)$')
ax[0].set_xlabel(r'$\gamma$')

sns.heatmap(r_acc_data_phi_1e_4, annot=True, cmap='Greens', xticklabels=gamma_vals_str, yticklabels=conditions, ax=ax[1],cbar_kws={'orientation': 'horizontal'})
ax[1].set_title(r'R Accuracy ($\phi$=1e-04)')
ax[1].set_xlabel(r'$\gamma$')

plt.tight_layout()

plt.savefig('ablation_experiment_phi_1e04_heatmap.png', dpi=300)

plt.show()

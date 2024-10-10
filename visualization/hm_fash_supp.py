import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

gamma_vals = [0, 0.0001, 0.1, 1]
lambda_vals = [0, 0.0001, 0.001, 0.01, 0.1]

gamma_vals_str = ['0', '1e-4', '1e-1', '1']
lambda_vals_str = ['0', '1e-4', '1e-2', '1e-1', '0.1']

forget_acc_new = np.array([
    [0.058, 0.084, 0.106, 0.104],
    [0.257, 0.136, 0.107, 0.104],
    [0.331, 0.293, 0.109, 0.103],
    [0.391, 0.388, 0.114, 0.105],
    [0.440, 0.443, 0.191, 0.106]
])

remain_acc_new = np.array([
    [0.870, 0.890, 0.914, 0.912],
    [0.862, 0.868, 0.914, 0.912],
    [0.859, 0.867, 0.913, 0.912],
    [0.831, 0.843, 0.904, 0.912],
    [0.820, 0.832, 0.844, 0.903]
])

font = 16

def calculate_distance(remain_acc, forget_acc, retrain_r, retrain_f):
    return np.sqrt((remain_acc - retrain_r) ** 2 + (forget_acc - retrain_f) ** 2)

retrain_r_new = 0.942  
retrain_f_new = 0.0    

distance_new = calculate_distance(remain_acc_new, forget_acc_new, retrain_r_new, retrain_f_new)

fig, ax = plt.subplots(1, 3, figsize=(12, 3))

sns.heatmap(remain_acc_new, annot=True, cmap='Greens', xticklabels=gamma_vals_str, yticklabels=lambda_vals_str, ax=ax[0])
ax[0].set_title('R Accuracy')
ax[0].set_xlabel(r'$\gamma, \uparrow$', fontsize=font)  
ax[0].set_ylabel(r'$\lambda$', fontsize=font, rotation=0, labelpad=20) 
ax[0].text(-0.1, 1.05, 'A', transform=ax[0].transAxes, fontsize=16, fontweight='bold', va='top')

sns.heatmap(forget_acc_new, annot=True, cmap='Reds_r', xticklabels=gamma_vals_str, yticklabels=lambda_vals_str, ax=ax[1])
ax[1].set_title('F Accuracy')
ax[1].set_xlabel(r'$\gamma, \downarrow$', fontsize=font) 
ax[1].text(-0.1, 1.05, 'B', transform=ax[1].transAxes, fontsize=16, fontweight='bold', va='top')

sns.heatmap(distance_new, annot=True, cmap='coolwarm_r', xticklabels=gamma_vals_str, yticklabels=lambda_vals_str, ax=ax[2])
ax[2].set_title('FashionMNIST: Dist. to Retraining')
ax[2].set_xlabel(r'$\gamma, \downarrow$', fontsize=font)
ax[2].text(-0.1, 1.05, 'C', transform=ax[2].transAxes, fontsize=16, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('heatmap_fashion_supp.png', dpi=300)
plt.show()

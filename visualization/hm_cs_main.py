
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

gamma_vals = [0, 0.0001, 0.1, 1]
lambda_vals = [0, 0.0001, 0.001, 0.01, 0.1]

gamma_vals_str = ['0', '1e-4', '1e-1', '1']
lambda_vals_str = ['0', '1e-4', '1e-2', '1e-1', '0.1']

forget_acc_medmnist = np.array([
    [0.016, 0.025, 0.016, 0.009],
    [0.016, 0.019, 0.016, 0.013],
    [0.016, 0.016, 0.013, 0.013],
    [0.016, 0.019, 0.016, 0.013],
    [0.016, 0.019, 0.016, 0.016]
])

remain_acc_medmnist = np.array([
    [0.661, 0.721, 0.688, 0.699],
    [0.666, 0.671, 0.688, 0.702],
    [0.668, 0.673, 0.686, 0.701],
    [0.669, 0.682, 0.685, 0.700],
    [0.671, 0.686, 0.656, 0.689]
])

forget_acc_cifar = np.array([
    [0.089, 0.115, 0.094, 0.096],
    [0.255, 0.195, 0.092, 0.096],
    [0.301, 0.347, 0.091, 0.096],
    [0.348, 0.409, 0.094, 0.097],
    [0.367, 0.421, 0.179, 0.101]
])

remain_acc_cifar = np.array([
    [0.837, 0.814, 0.810, 0.810],
    [0.868, 0.842, 0.810, 0.810],
    [0.876, 0.877, 0.809, 0.809],
    [0.877, 0.878, 0.810, 0.809],
    [0.875, 0.876, 0.852, 0.810]
])

def calculate_distance(remain_acc, forget_acc, retrain_r, retrain_f):
    return np.sqrt((remain_acc - retrain_r) ** 2 + (forget_acc - retrain_f) ** 2)


retrain_r_medmnist = 0.975  
retrain_f_medmnist = 0.0    

retrain_r_cifar = 0.899     
retrain_f_cifar = 0.0       

distance_medmnist = calculate_distance(remain_acc_medmnist, forget_acc_medmnist, retrain_r_medmnist, retrain_f_medmnist)
distance_cifar = calculate_distance(remain_acc_cifar, forget_acc_cifar, retrain_r_cifar, retrain_f_cifar)

fig, ax = plt.subplots(2, 3, figsize=(12, 6))
font = 16


sns.heatmap(remain_acc_medmnist, annot=True, cmap='Greens', xticklabels=gamma_vals_str, yticklabels=lambda_vals_str, ax=ax[0, 0])
ax[0, 0].set_title('MedMNIST: R Accuracy')
ax[0, 0].set_xlabel(r'$\gamma, \uparrow$', fontsize=font) 
ax[0, 0].set_ylabel(r'$\lambda$', fontsize=font, rotation=0, labelpad=20) 
ax[0, 0].text(-0.1, 1.05, 'A', transform=ax[0, 0].transAxes, fontsize=16, fontweight='bold', va='top')

sns.heatmap(forget_acc_medmnist, annot=True, cmap='Reds_r', xticklabels=gamma_vals_str, yticklabels=lambda_vals_str, ax=ax[0, 1])
ax[0, 1].set_title('MedMNIST: F Accuracy')
ax[0, 1].set_xlabel(r'$\gamma, \downarrow$', fontsize=font)  
ax[0, 1].text(-0.1, 1.05, 'B', transform=ax[0, 1].transAxes, fontsize=16, fontweight='bold', va='top')

sns.heatmap(distance_medmnist, annot=True, cmap='coolwarm_r', xticklabels=gamma_vals_str, yticklabels=lambda_vals_str, ax=ax[0, 2])
ax[0, 2].set_title('MedMNIST: Dist. to Retraining')
ax[0, 2].set_xlabel(r'$\gamma, \downarrow$', fontsize=font)  
ax[0, 2].text(-0.1, 1.05, 'C', transform=ax[0, 2].transAxes, fontsize=16, fontweight='bold', va='top')

sns.heatmap(remain_acc_cifar, annot=True, cmap='Greens', xticklabels=gamma_vals_str, yticklabels=lambda_vals_str, ax=ax[1, 0])
ax[1, 0].set_title('CIFAR-10: R Accuracy')
ax[1, 0].set_xlabel(r'$\gamma,\uparrow$', fontsize=font) 
ax[1, 0].set_ylabel(r'$\lambda$', fontsize=font, rotation=0, labelpad=20)  
ax[1, 0].text(-0.1, 1.05, 'D', transform=ax[1, 0].transAxes, fontsize=16, fontweight='bold', va='top')

sns.heatmap(forget_acc_cifar, annot=True, cmap='Reds_r', xticklabels=gamma_vals_str, yticklabels=lambda_vals_str, ax=ax[1, 1])
ax[1, 1].set_title('CIFAR-10: F Accuracy')
ax[1, 1].set_xlabel(r'$\gamma, \downarrow$', fontsize=font)  
ax[1, 1].text(-0.1, 1.05, 'E', transform=ax[1, 1].transAxes, fontsize=16, fontweight='bold', va='top')

sns.heatmap(distance_cifar, annot=True, cmap='coolwarm_r', xticklabels=gamma_vals_str, yticklabels=lambda_vals_str, ax=ax[1, 2])
ax[1, 2].set_title('CIFAR-10: Dist. to Retraining')
ax[1, 2].set_xlabel(r'$\gamma, \downarrow$', fontsize=font)
ax[1, 2].text(-0.1, 1.05, 'F', transform=ax[1, 2].transAxes, fontsize=16, fontweight='bold', va='top')



# Adjust layout and save with high resolution
plt.tight_layout()
plt.savefig('heatmap_cs_main.png', dpi=300)
plt.show()

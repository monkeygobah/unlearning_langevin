import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

gamma_vals = [0, 1e-4, 1e-1, 1]
lambda_vals = [0, 1e-4, 1e-2, 1e-1, 0.1]

gamma_vals_str = ['0', '1e-4', '1e-1', '1']
lambda_vals_str = ['0', '1e-4', '1e-2', '1e-1', '0.1']

forget_acc_fundus = np.array([
    [0.094, 0.052, 0.042, 0.052],
    [0.479, 0.052, 0.042, 0.052],
    [0.625, 0.995, 0.042, 0.052],
    [0.740, 0.625, 0.042, 0.052],
    [0.740, 0.615, 0.052, 0.052]
])

remain_acc_fundus = np.array([
    [0.995, 0.984, 0.990, 0.984375],
    [0.995, 0.984, 0.990, 0.984],
    [0.995, 1.000, 0.990, 0.984],
    [0.990, 1.000, 0.990, 0.984],
    [0.990, 1.000, 0.990, 0.984]
])

forget_acc_mri = np.array([
    [0.016, 0.025, 0.016, 0.009],
    [0.016, 0.019, 0.016, 0.013],
    [0.016, 0.016, 0.013, 0.013],
    [0.016, 0.019, 0.016, 0.013],
    [0.016, 0.019, 0.016, 0.016]
])

remain_acc_mri = np.array([
    [0.661, 0.721, 0.688, 0.699],
    [0.666, 0.671, 0.688, 0.702],
    [0.668, 0.673, 0.686, 0.701],
    [0.669, 0.682, 0.685, 0.700],
    [0.671, 0.686, 0.656, 0.689]
])

def calculate_distance(remain_acc, forget_acc, retrain_r, retrain_f):
    return np.sqrt((remain_acc - retrain_r) ** 2 + (forget_acc - retrain_f) ** 2)

retrain_r_fundus = 1.0  
retrain_f_fundus = 0.0  

retrain_r_mri = 0.975  
retrain_f_mri = 0.0  

distance_fundus = calculate_distance(remain_acc_fundus, forget_acc_fundus, retrain_r_fundus, retrain_f_fundus)
distance_mri = calculate_distance(remain_acc_mri, forget_acc_mri, retrain_r_mri, retrain_f_mri)

fig, ax = plt.subplots(2, 3, figsize=(12, 6))

font = 16

sns.heatmap(remain_acc_fundus, annot=True, cmap='Greens', xticklabels=gamma_vals_str, yticklabels=lambda_vals_str, ax=ax[0, 0])
ax[0, 0].set_title('CFP: R Accuracy')
ax[0, 0].set_xlabel(r'$\gamma, \uparrow$', fontsize=font)  
ax[0, 0].set_ylabel(r'$\lambda$', fontsize=font, rotation=0, labelpad=20)  
ax[0, 0].text(-0.1, 1.05, 'A', transform=ax[0, 0].transAxes, fontsize=16, fontweight='bold', va='top')

sns.heatmap(forget_acc_fundus, annot=True, cmap='Reds_r', xticklabels=gamma_vals_str, yticklabels=lambda_vals_str, ax=ax[0, 1])
ax[0, 1].set_title('CFP: F Accuracy')
ax[0, 1].set_xlabel(r'$\gamma, \downarrow$', fontsize=font)  
ax[0, 1].text(-0.1, 1.05, 'B', transform=ax[0, 1].transAxes, fontsize=16, fontweight='bold', va='top')

sns.heatmap(distance_fundus, annot=True, cmap='coolwarm_r', xticklabels=gamma_vals_str, yticklabels=lambda_vals_str, ax=ax[0, 2])
ax[0, 2].set_title('CFP: Dist. to Retraining')
ax[0, 2].set_xlabel(r'$\gamma, \downarrow$', fontsize=font) 
ax[0, 2].text(-0.1, 1.05, 'C', transform=ax[0, 2].transAxes, fontsize=16, fontweight='bold', va='top')

sns.heatmap(remain_acc_mri, annot=True, cmap='Greens', xticklabels=gamma_vals_str, yticklabels=lambda_vals_str, ax=ax[1, 0])
ax[1, 0].set_title('MRI: R Accuracy')
ax[1, 0].set_xlabel(r'$\gamma,\uparrow$', fontsize=font)  
ax[1, 0].set_ylabel(r'$\lambda$', fontsize=font, rotation=0, labelpad=20) 
ax[1, 0].text(-0.1, 1.05, 'D', transform=ax[1, 0].transAxes, fontsize=16, fontweight='bold', va='top')

sns.heatmap(forget_acc_mri, annot=True, cmap='Reds_r', xticklabels=gamma_vals_str, yticklabels=lambda_vals_str, ax=ax[1, 1])
ax[1, 1].set_title('MRI: F Accuracy')
ax[1, 1].set_xlabel(r'$\gamma, \downarrow$', fontsize=font) 
ax[1, 1].text(-0.1, 1.05, 'E', transform=ax[1, 1].transAxes, fontsize=16, fontweight='bold', va='top')

sns.heatmap(distance_mri, annot=True, cmap='coolwarm_r', xticklabels=gamma_vals_str, yticklabels=lambda_vals_str, ax=ax[1, 2])
ax[1, 2].set_title('MRI: Dist. to Retraining')
ax[1, 2].set_xlabel(r'$\gamma, \downarrow$', fontsize=font) 
ax[1, 2].text(-0.1, 1.05, 'F', transform=ax[1, 2].transAxes, fontsize=16, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('heatmap_med_main.png', dpi=300)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

epochs = [0, 1, 2, 5]
f_acc_hybrid1 = [0.176, 0.356, 0.251, 0.056]
r_acc_hybrid1 = [0.863, 0.912, 0.912, 0.916]
f_acc_hybrid2 = [0, 0, 0, 0]
r_acc_hybrid2 = [0.234, 0.907, 0.905, 0.913]

original_f_acc_hybrid1 = 0.089
original_r_acc_hybrid1 = 0.878

original_f_acc_hybrid2 = 0.0001
original_r_acc_hybrid2 = 0.890

fig, ax = plt.subplots(1, 2, figsize=(6, 3))

ax[0].plot(epochs, f_acc_hybrid1, label='Composition F Acc', color='blue')
ax[0].plot(epochs, r_acc_hybrid1, label='Composition R Acc', color='red')
ax[0].axhline(y=original_f_acc_hybrid1, color='blue', linestyle='--', label='Original F Acc')
ax[0].axhline(y=original_r_acc_hybrid1, color='red', linestyle='--', label='Original R Acc')
ax[0].set_title('Composition 1')
ax[0].set_xlabel('Finetune Epochs')
ax[0].set_ylabel('Accuracy')

ax[1].plot(epochs, f_acc_hybrid2, label='Composition F Acc', color='blue')
ax[1].plot(epochs, r_acc_hybrid2, label='Composition R Acc', color='red')
ax[1].axhline(y=original_f_acc_hybrid2, color='blue', linestyle='--', label='Original F Acc')
ax[1].axhline(y=original_r_acc_hybrid2, color='red', linestyle='--', label='Original R Acc')
ax[1].set_title('Composition 2')
ax[1].set_xlabel('Finetune Epochs')
ax[1].legend()

# Adjust layout and display
plt.tight_layout()
plt.savefig('hybrid_1_and_2_accuracy.png', dpi=300)
plt.show()

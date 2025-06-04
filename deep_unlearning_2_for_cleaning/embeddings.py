import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm

from matplotlib.colors import ListedColormap

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import shutil
from torchvision.utils import save_image
import torchvision.transforms.functional as F

def get_embeddings_predictions_and_forget_indications(model, forget_loader, remain_loader, device):
    model.eval()
    embeddings = []
    predictions = []
    is_forget_sample = []
    true_labels = []

    with torch.no_grad():
        for data, target in tqdm(forget_loader, desc="Processing Forget Samples"):
            data = data.to(device)
            if isinstance(model, torch.nn.DataParallel):
                emb = model.module.get_embedding(data)
            else:
                emb = model.get_embedding(data)            
            output = model(data)
            embeddings.append(emb.cpu().numpy())
            predictions.append(output.argmax(dim=1).cpu().numpy())
            true_labels.append(target.cpu().numpy())
            is_forget_sample.append(np.ones(len(data), dtype=bool))

        for data, target in tqdm(remain_loader, desc="Processing Remain Samples"):
            data = data.to(device)
            if isinstance(model, torch.nn.DataParallel):
                emb = model.module.get_embedding(data)
            else:
                emb = model.get_embedding(data)
            output = model(data)
            embeddings.append(emb.cpu().numpy())
            predictions.append(output.argmax(dim=1).cpu().numpy())
            true_labels.append(target.cpu().numpy())
            is_forget_sample.append(np.zeros(len(data), dtype=bool))

    embeddings = np.concatenate(embeddings, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    is_forget_sample = np.concatenate(is_forget_sample, axis=0)

    return embeddings, predictions, is_forget_sample, true_labels



def plot_tsne(embeddings, predictions, is_forget_sample, true_labels, title, ax, add_legend=False):
    tsne = TSNE(n_components=2, random_state=0)
    pts = tsne.fit_transform(embeddings)

    mis = predictions != true_labels
    rem = ~is_forget_sample
    fog =  is_forget_sample

    # Color maps
    class_colors = {0:'red', 1:'green', 2:'blue'}  # Extend as needed
    true_color_map = np.array([class_colors[t] for t in true_labels])
    pred_color_map = np.array([class_colors[p] for p in predictions])
    edge_colors = np.where(mis, pred_color_map, 'none')
    line_widths = np.where(mis, 3, 0)

    # === Draw tighter, smoother decision boundary ===
    def draw_boundary():
        clf = KNeighborsClassifier(n_neighbors=5).fit(pts, predictions)
        x_min, x_max = pts[:,0].min(), pts[:,0].max()
        y_min, y_max = pts[:,1].min(), pts[:,1].max()
        padding = 2.0
        xx, yy = np.meshgrid(np.linspace(x_min-padding, x_max+padding, 400),
                             np.linspace(y_min-padding, y_max+padding, 400))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        cmap = ListedColormap([class_colors[c] for c in sorted(class_colors)])
        ax.contourf(xx, yy, Z, alpha=0.25, cmap=cmap, 
                    levels=np.arange(len(class_colors)+1)-0.5)

        # Set limits tightly around data
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

    draw_boundary()

    # === Plot remain samples ===
    ax.scatter(
        pts[rem,0], pts[rem,1],
        facecolors=true_color_map[rem],
        edgecolors=edge_colors[rem],
        linewidths=line_widths[rem],
        marker='o', s=200, alpha=0.8
    )

    # === Plot forget samples ===
    ax.scatter(
        pts[fog,0], pts[fog,1],
        facecolors=true_color_map[fog],
        edgecolors=edge_colors[fog],
        linewidths=line_widths[fog],
        marker='*', s=500, alpha=0.95
    )

    # === Style ===
    ax.set_title(title, fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])

    if add_legend:
        handles = [
            plt.Line2D([0],[0], marker='o', color='w',
                       markerfacecolor=class_colors[c], markersize=10)
            for c in sorted(class_colors)
        ]
        labels = [f'Class {c}' for c in sorted(class_colors)]
        star = plt.Line2D([0],[0], marker='*', color='k',
                          markerfacecolor='white', markersize=12,
                          linestyle='None', label='Forget Set')
        mis_edge = plt.Line2D([0],[0], marker='o', markerfacecolor='white',
                              markeredgecolor='grey', markersize=10,
                              linestyle='None', label='Misclassified Edge')
        ax.legend(handles + [star, mis_edge],
                  labels + ['Forget Set'],
                  loc='best', fontsize=11, frameon=True)


def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return tensor * std + mean



def save_unlearning_examples(model, forget_loader, remain_loader, device, save_dir):
    print('getting exampels imgaes')
    model.eval()
    os.makedirs(f"{save_dir}/forget_misclassified", exist_ok=True)
    os.makedirs(f"{save_dir}/remain_correct", exist_ok=True)

    with torch.no_grad():
        # Process forget set
        for i, (data, target) in enumerate(forget_loader):
            data = data.to(device)
            output = model(data)
            preds = output.argmax(dim=1).cpu()
            target = target.cpu()

            for j in range(len(data)):
                if preds[j] != target[j]:
                    path = f"{save_dir}/forget_misclassified/img_{i}_{j}_pred{preds[j].item()}_true{target[j].item()}.png"
                    img = unnormalize(data[j].cpu()).clamp(0, 1)
                    save_image(img, path)

        # Process remain set
        for i, (data, target) in enumerate(remain_loader):
            data = data.to(device)
            output = model(data)
            preds = output.argmax(dim=1).cpu()
            target = target.cpu()

            for j in range(len(data)):
                if preds[j] == target[j]:
                    path = f"{save_dir}/remain_correct/img_{i}_{j}_pred{preds[j].item()}_true{target[j].item()}.png"
                    img = unnormalize(data[j].cpu()).clamp(0, 1)
                    save_image(img, path)




# def plot_tsne(embeddings, predictions, is_forget_sample, title, ax, add_legend=False):
#     tsne = TSNE(n_components=2, random_state=0)
#     tsne_result = tsne.fit_transform(embeddings)
    
#     unique_classes = np.unique(predictions)
#     class_colors = {0: 'red', 1: 'green', 2: 'blue'}
#     colors = [class_colors[cls] for cls in unique_classes]
#     custom_cmap = ListedColormap(colors)

#     # class_colors = ['red', 'green', 'blue'][:len(unique_classes)]
#     # custom_cmap = ListedColormap(class_colors)
#     scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], c=predictions, cmap=custom_cmap, s=100, alpha=0.7)
    
#     forget_tsne = tsne_result[is_forget_sample]
#     forget_preds = predictions[is_forget_sample]
#     ax.scatter(forget_tsne[:, 0], forget_tsne[:, 1], c=forget_preds, cmap=custom_cmap, s=400, edgecolors='black', marker='*')
    
#     ax.set_title(title)
#     ax.set_xticks([])
#     ax.set_yticks([])
    
#     if add_legend:
#         handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_cmap(i), markersize=10) for i in range(len(unique_classes))]
#         labels = [f'Class {cls}' for cls in unique_classes]
#         legend1 = ax.legend(handles, labels, title="Classes", loc="best", fontsize=12)
#         ax.add_artist(legend1)


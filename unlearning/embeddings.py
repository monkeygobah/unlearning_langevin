import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm

from matplotlib.colors import ListedColormap

def get_embeddings_predictions_and_forget_indications(model, forget_loader, remain_loader, device):
    model.eval()
    embeddings = []
    predictions = []
    is_forget_sample = []
    
    with torch.no_grad():
        for data, target in tqdm(forget_loader, desc="Processing Forget Samples"):
            data = data.to(device)
            output = model(data)
            embeddings.append(output.cpu().numpy())
            preds = output.argmax(dim=1)
            predictions.append(preds.cpu().numpy())
            is_forget_sample.append(True)
        
        for data, target in tqdm(remain_loader, desc="Processing Remain Samples"):
            data = data.to(device)
            output = model(data)
            embeddings.append(output.cpu().numpy())
            preds = output.argmax(dim=1)
            predictions.append(preds.cpu().numpy())
            is_forget_sample.append(False)
    
    embeddings = np.concatenate(embeddings, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    is_forget_sample = np.array(is_forget_sample)
    
    return embeddings, predictions, is_forget_sample

def plot_tsne(embeddings, predictions, is_forget_sample, title, ax, add_legend=False):
    tsne = TSNE(n_components=2, random_state=0)
    tsne_result = tsne.fit_transform(embeddings)
    
    unique_classes = np.unique(predictions)
    class_colors = {0: 'red', 1: 'green', 2: 'blue'}
    colors = [class_colors[cls] for cls in unique_classes]
    custom_cmap = ListedColormap(colors)

    # class_colors = ['red', 'green', 'blue'][:len(unique_classes)]
    # custom_cmap = ListedColormap(class_colors)
    scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], c=predictions, cmap=custom_cmap, s=100, alpha=0.7)
    
    forget_tsne = tsne_result[is_forget_sample]
    forget_preds = predictions[is_forget_sample]
    ax.scatter(forget_tsne[:, 0], forget_tsne[:, 1], c=forget_preds, cmap=custom_cmap, s=400, edgecolors='black', marker='*')
    
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    
    if add_legend:
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_cmap(i), markersize=10) for i in range(len(unique_classes))]
        labels = [f'Class {cls}' for cls in unique_classes]
        legend1 = ax.legend(handles, labels, title="Classes", loc="best", fontsize=12)
        ax.add_artist(legend1)


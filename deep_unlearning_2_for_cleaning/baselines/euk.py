from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# from mia_metric import *
from baseline_utils import *

from thirdparty.repdistiller.helper.util import adjust_learning_rate as sgda_adjust_learning_rate
from thirdparty.repdistiller.distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from thirdparty.repdistiller.distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss

from thirdparty.repdistiller.helper.loops import train_distill, train_distill_hide, train_distill_linear, train_vanilla, train_negrad, train_bcu, train_bcu_distill, validate
from thirdparty.repdistiller.helper.pretrain import init

import copy


def replace_loader_dataset(data_loader, dataset, batch_size=128, seed=1, shuffle=True):
    seed_torch(seed)
    loader_args = {'num_workers': 0, 'pin_memory': False}
    def _init_fn(worker_id):
        np.random.seed(int(seed))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,num_workers=0,pin_memory=True,shuffle=shuffle)


def cfk_unlearn(model_cfk, r_loader, model_name):
    lr_decay_epochs = [10,15,20]
    cfk_lr = 0.01
    cfk_epochs = 10

    # cfk_bs = 64
    # r_loader = replace_loader_dataset(train_loader_full,retain_dataset, seed=seed, batch_size=cfk_bs, shuffle=True)
    # model_cfk = copy.deepcopy(model)

    for param in model_cfk.parameters():
        param.requires_grad_(False)

    if model_name == 'allcnn':
        layers = [9]
        for k in layers:
            for param in model_cfk.features[k].parameters():
                param.requires_grad_(True)

    elif model_name == "resnet":
        for param in model_cfk.resnet_base.layer4.parameters():
            param.requires_grad_(True)

    else:
        raise NotImplementedError


    fk_fientune(model_cfk, r_loader, lr_decay_epochs, epochs=cfk_epochs, quiet=False, lr=cfk_lr)

    return model_cfk



# def euk_unlearn(model, r_loader, model_name):
#     """
#     Roll back selected parameters to their original state (model_initial),
#     then fine-tune them on remain data (r_loader).
#     """

#     lr_decay_epochs = [10, 15, 20]
#     euk_lr = 0.01
#     euk_epochs = 10
#     euk_bs = 64


#     model_initial = copy.deepcopy(model)
#     model_euk = copy.deepcopy(model)

#     # 1) Freeze everything
#     for param in model_euk.parameters():
#         param.requires_grad_(False)

#     if model_name == 'allcnn':
#         layers = [9]

#         with torch.no_grad():
#             for k in layers:
#                 block_module = model_euk.features[k]
#                 block_module_init = model_initial.features[k]
#                 for i, sublayer in enumerate(block_module):
#                     if hasattr(sublayer, 'weight') and hasattr(block_module_init[i], 'weight'):
#                         sublayer.weight.copy_(block_module_init[i].weight)
#                     if hasattr(sublayer, 'bias') and hasattr(block_module_init[i], 'bias'):
#                         sublayer.bias.copy_(block_module_init[i].bias)

#             model_euk.classifier[0].weight.copy_(model_initial.classifier[0].weight)
#             model_euk.classifier[0].bias.copy_(model_initial.classifier[0].bias)

#         for k in layers:
#             for param in model_euk.features[k].parameters():
#                 param.requires_grad_(True)



#     elif model_name == 'resnet':
#         with torch.no_grad():
#             for i in range(2):
#                 block_euk = model_euk.resnet_base.layer4[i]
#                 block_init = model_initial.resnet_base.layer4[i]

#                 # BN1
#                 if hasattr(block_euk.bn1, 'weight'):
#                     block_euk.bn1.weight.copy_(block_init.bn1.weight)
#                 if hasattr(block_euk.bn1, 'bias'):
#                     block_euk.bn1.bias.copy_(block_init.bn1.bias)

#                 # Conv1
#                 if hasattr(block_euk.conv1, 'weight'):
#                     block_euk.conv1.weight.copy_(block_init.conv1.weight)
#                 if hasattr(block_euk.conv1, 'bias'):
#                     block_euk.conv1.bias.copy_(block_init.conv1.bias)

#                 # BN2
#                 if hasattr(block_euk.bn2, 'weight'):
#                     block_euk.bn2.weight.copy_(block_init.bn2.weight)
#                 if hasattr(block_euk.bn2, 'bias'):
#                     block_euk.bn2.bias.copy_(block_init.bn2.bias)

#                 # Conv2
#                 if hasattr(block_euk.conv2, 'weight'):
#                     block_euk.conv2.weight.copy_(block_init.conv2.weight)
#                 if hasattr(block_euk.conv2, 'bias'):
#                     block_euk.conv2.bias.copy_(block_init.conv2.bias)

#             # Example of rolling back a shortcut
#             if hasattr(model_euk.resnet_base.layer4[0], 'downsample') or hasattr(model_euk.resnet_base.layer4[0], 'shortcut'):
#                 # Suppose it's named 'downsample' in newer PyTorch or 'shortcut' in older
#                 # Adjust accordingly to your resnet impl
#                 if hasattr(model_euk.resnet_base.layer4[0], 'shortcut'):
#                     model_euk.resnet_base.layer4[0].shortcut[0].weight.copy_(
#                         model_initial.resnet_base.layer4[0].shortcut[0].weight
#                     )
#                 elif hasattr(model_euk.resnet_base.layer4[0], 'downsample'):
#                     model_euk.resnet_base.layer4[0].downsample[0].weight.copy_(
#                         model_initial.resnet_base.layer4[0].downsample[0].weight
#                     )

#         for param in model_euk.resnet_base.layer4.parameters():
#             param.requires_grad_(True)

#     else:
#         raise NotImplementedError(f"EU-K unlearning is not implemented for {model_name}.")

#     # 3) Fine-tune the partially restored layers
#     fk_fientune(model_euk, r_loader, epochs=euk_epochs, quiet=True, lr=euk_lr)
#     return model_euk



def euk_unlearn(model, r_loader, model_name):
    lr_decay_epochs = [10,15,20]
    euk_lr = 0.01
    euk_epochs = 10
    euk_bs = 64
    model_initial = model
    # r_loader = replace_loader_dataset(train_loader_full, retain_dataset, seed=seed, batch_size=euk_bs, shuffle=True)
    model_euk = copy.deepcopy(model)

    for param in model_euk.parameters():
        param.requires_grad_(False)

    if model_name == 'allcnn':
        layers = [9]

        with torch.no_grad():
            for k in layers:
                for i in range(0,3):
                    try:
                        model_euk.features[k][i].weight.copy_(model_initial.features[k][i].weight)
                    except:
                        print ("block {}, layer {} does not have weights".format(k,i))
                    try:
                        model_euk.features[k][i].bias.copy_(model_initial.features[k][i].bias)
                    except:
                        print ("block {}, layer {} does not have bias".format(k,i))
            model_euk.classifier[0].weight.copy_(model_initial.classifier[0].weight)
            model_euk.classifier[0].bias.copy_(model_initial.classifier[0].bias)

        for k in layers:
            for param in model_euk.features[k].parameters():
                param.requires_grad_(True)

    elif model_name == "resnet":
        with torch.no_grad():
            for i in range(0,2):
                try:
                    model_euk.resnet_base.layer4[i].bn1.weight.copy_(model_initial.resnet_base.layer4[i].bn1.weight)
                except:
                    print ("block 4, layer {} does not have weight".format(i))
                try:
                    model_euk.resnet_base.layer4[i].bn1.bias.copy_(model_initial.resnet_base.layer4[i].bn1.bias)
                except:
                    print ("block 4, layer {} does not have bias".format(i))
                try:
                    model_euk.resnet_base.layer4[i].conv1.weight.copy_(model_initial.resnet_base.layer4[i].conv1.weight)
                except:
                    print ("block 4, layer {} does not have weight".format(i))
                try:
                    model_euk.resnet_base.layer4[i].conv1.bias.copy_(model_initial.resnet_base.layer4[i].conv1.bias)
                except:
                    print ("block 4, layer {} does not have bias".format(i))

                try:
                    model_euk.resnet_base.layer4[i].bn2.weight.copy_(model_initial.resnet_base.layer4[i].bn2.weight)
                except:
                    print ("block 4, layer {} does not have weight".format(i))
                try:
                    model_euk.resnet_base.layer4[i].bn2.bias.copy_(model_initial.resnet_base.layer4[i].bn2.bias)
                except:
                    print ("block 4, layer {} does not have bias".format(i))
                try:
                    model_euk.resnet_base.layer4[i].conv2.weight.copy_(model_initial.resnet_base.layer4[i].conv2.weight)
                except:
                    print ("block 4, layer {} does not have weight".format(i))
                try:
                    model_euk.resnet_base.layer4[i].conv2.bias.copy_(model_initial.resnet_base.layer4[i].conv2.bias)
                except:
                    print ("block 4, layer {} does not have bias".format(i))

            # model_euk.resnet_base.layer4[0].shortcut[0].weight.copy_(model_initial.resnet_base.layer4[0].shortcut[0].weight)

            if hasattr(model_euk.resnet_base.layer4[0], 'downsample') and model_euk.resnet_base.layer4[0].downsample is not None:
                model_euk.resnet_base.layer4[0].downsample[0].weight.copy_(
                    model_initial.resnet_base.layer4[0].downsample[0].weight
                )


        for param in model_euk.resnet_base.layer4.parameters():
            param.requires_grad_(True)

    else:
        raise NotImplementedError


    fk_fientune(model_euk, r_loader,lr_decay_epochs, epochs=euk_epochs, quiet=True, lr=euk_lr)
    return model_euk



def fk_fientune(model, data_loader,lr_decay_epochs, lr=0.01, epochs=10, quiet=False):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0)
    for epoch in range(epochs):
        sgda_adjust_learning_rate(epoch, optimizer,lr_decay_epochs)
        train_vanilla(epoch, data_loader, model, loss_fn, optimizer)
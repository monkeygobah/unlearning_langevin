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
import torch.optim as optim

# from mia_metric import *
from thirdparty.repdistiller.helper.util import adjust_learning_rate as sgda_adjust_learning_rate
from thirdparty.repdistiller.distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from thirdparty.repdistiller.distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss

from thirdparty.repdistiller.helper.loops import train_distill, train_distill_hide, train_distill_linear, train_vanilla, train_negrad, train_bcu, train_bcu_distill, validate
from thirdparty.repdistiller.helper.pretrain import init

import copy
import time

def scrub_met(teacher, student, remain_loader, forget_loader,model_name,dataset,seed=2022):
    optim_name  = 'sgd'
    gamma = 1
    alpha = 0.5
    beta = 0
    smoothing = 0.5
    msteps = 3
    clip = 0.2
    sstart = 10
    kd_T = 4
    distill = 'kd'

    sgda_epochs = 5
    sgda_learning_rate = 0.0005
    lr_decay_epochs = [3,5,9]
    lr_decay_rate = 0.1
    sgda_weight_decay = 5e-4
    sgda_momentum = 0.9

    model_t = teacher
    model_s = student


    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(kd_T)
    criterion_kd = DistillKL(kd_T)


    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    if optim_name  == "sgd":
        optimizer = optim.SGD(trainable_list.parameters(),
                              lr=sgda_learning_rate,
                              momentum=sgda_momentum,
                              weight_decay=sgda_weight_decay)
    # elif optim == "adam": 
    #     optimizer = optim.Adam(trainable_list.parameters(),
    #                           lr=sgda_learning_rate,
    #                           weight_decay=sgda_weight_decay)
    # elif optim == "rmsp":
    #     optimizer = optim.RMSprop(trainable_list.parameters(),
    #                           lr=sgda_learning_rate,
    #                           momentum=sgda_momentum,
    #                           weight_decay=sgda_weight_decay)

    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True


    t1 = time.time()
    acc_rs = []
    acc_fs = []
    acc_vs = []
    acc_fvs = []
    
    
    # forget_validation_loader = copy.deepcopy(valid_loader_full)
    # fgt_cls = list(np.unique(forget_loader.dataset.targets))
    # indices = [i in fgt_cls for i in forget_validation_loader.dataset.targets]
    # forget_validation_loader.dataset.data = forget_validation_loader.dataset.data[indices]
    # forget_validation_loader.dataset.targets = forget_validation_loader.dataset.targets[indices]
    
    scrub_name = "checkpoints/scrub_{}_{}_seed{}_step".format(model_name, dataset, seed)
    for epoch in range(1, sgda_epochs + 1):

        lr = sgda_adjust_learning_rate(epoch, optimizer,lr_decay_epochs)

        acc_r, acc5_r, loss_r = validate(remain_loader, model_s, criterion_cls,  True)
        acc_f, acc5_f, loss_f = validate(forget_loader, model_s, criterion_cls,  True)
        # acc_v, acc5_v, loss_v = validate(valid_loader_full, model_s, criterion_cls, args, True)
        # acc_fv, acc5_fv, loss_fv = validate(forget_validation_loader, model_s, criterion_cls, args, True)
        acc_rs.append(100-acc_r.item())
        acc_fs.append(100-acc_f.item())
        # acc_vs.append(100-acc_v.item())
        # acc_fvs.append(100-acc_fv.item())

        maximize_loss = 0
        if epoch <= msteps:
            maximize_loss = train_distill(epoch, forget_loader, module_list, None, criterion_list, optimizer,  "maximize")
        
        
        train_acc, train_loss = train_distill(epoch, remain_loader, module_list, None, criterion_list, optimizer,  "minimize",)
        
        torch.save(model_s.state_dict(), scrub_name+str(epoch)+".pt")


        print ("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(maximize_loss, train_loss, train_acc))
    
    t2 = time.time()
    print (t2-t1)

    acc_r, acc5_r, loss_r = validate(remain_loader, model_s, criterion_cls,  True)
    acc_f, acc5_f, loss_f = validate(forget_loader, model_s, criterion_cls,  True)
    # acc_v, acc5_v, loss_v = validate(valid_loader_full, model_s, criterion_cls, args, True)
    # acc_fv, acc5_fv, loss_fv = validate(forget_validation_loader, model_s, criterion_cls, args, True)
    acc_rs.append(100-acc_r.item())
    acc_fs.append(100-acc_f.item())
    # acc_vs.append(100-acc_v.item())
    # acc_fvs.append(100-acc_fv.item())

    # from matplotlib import pyplot as plt
    # indices = list(range(0,len(acc_rs)))
    # plt.plot(indices, acc_rs, marker='*', color=u'#1f77b4', alpha=1, label='retain-set')
    # plt.plot(indices, acc_fs, marker='o', color=u'#ff7f0e', alpha=1, label='forget-set')
    # plt.plot(indices, acc_vs, marker='^', color=u'#2ca02c',alpha=1, label='validation-set')
    # plt.plot(indices, acc_fvs, marker='.', color='red',alpha=1, label='forget-validation-set')
    # plt.legend(prop={'size': 14})
    # plt.tick_params(labelsize=12)
    # plt.xlabel('epoch',size=14)
    # plt.ylabel('error',size=14)
    # plt.grid()
    # plt.show()
    
    
    try:
        selected_idx, _ = min(enumerate(acc_fs), key=lambda x: abs(x[1]-acc_fvs[-1]))
    except:
        selected_idx = len(acc_fs) - 1

    print ("the selected index is {}".format(selected_idx))
    selected_model = "checkpoints/scrub_{}_{}_seed{}_step{}.pt".format(model_name, dataset, seed, int(selected_idx))
    model_s_final = copy.deepcopy(model_s)
    model_s.load_state_dict(torch.load(selected_model))
    
    
    return model_s, model_s_final


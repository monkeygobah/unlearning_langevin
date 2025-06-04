import copy
import utils
from trainer import eval, loss_picker, optimizer_picker
import numpy as np
import torch
from torch import nn
from adv_generator import LinfPGD, inf_generator, FGSM
import tqdm
import time
from models import init_params as w_init
from expand_exp import curvature, weight_assign
import pandas as pd
from make_dataloaders import *


def save_accuracies_to_csv(gamma, test_acc, forget_acc, remain_acc, ul_class):
    # Define the filename based on the gamma value
    filename = f'{gamma}_class{ul_class}_sgld_accuracies.csv'
    
    # Create a DataFrame with the accuracies and gamma
    data = {
        'gamma': [gamma],
        'test_acc': [test_acc],
        'forget_acc': [forget_acc],
        'remain_acc': [remain_acc]
    }
    df = pd.DataFrame(data)
    
    df.to_csv(filename, index=False)


def boundary_shrink(ori_model, train_forget_loader, dt, dv, test_loader, device, 
                    bound=0.1, step=8 / 255, iter=5, poison_epoch=10, forget_class=0, path='./',
                    lambda_=0.7, bias=-0.5, slope=5.0, gamma = 0, dist = 'normal',
                    to_forget =None, custom_forget = False, test_metadata = None, train_metadata = None, output_name=None,
                    use_linfpgd = False, lamda = 0, l1_norm = False, data_name = None, scaling=None, oculoplastics=False,retrain_model=None,
                    train_remain_loader = None, use_logits = False, remain_reg_param = 0.0, logit_preprocess=False, selective_unlearning=False):
    
    start = time.time()
    if data_name == 'mnist':
        norm = False  # None#True if data_name != "mnist" else False
    else:
        norm = True

    random_start = False  # False if attack != "pgd" else True
    
    # Copy the original model to create a test model and an unlearning model, 
    
    test_model = copy.deepcopy(ori_model).to(device)

    unlearn_model = copy.deepcopy(ori_model).to(device)

    start_time = time.time()
    
    # Adversarial attack configuration
    if use_linfpgd:
        # only for ViT
        adv = LinfPGD(test_model, bound, step, iter, norm, random_start, device,  gamma)
    else:
        adv = FGSM(test_model, bound, norm, random_start, device)
    
    forget_data_gen = inf_generator(train_forget_loader)
    remain_data_gen = inf_generator(train_remain_loader)

    batches_per_epoch = len(train_forget_loader)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=0.00001, momentum=0.9)

    num_hits = 0
    num_sum = 0
    nearest_label = []

    num_iterations = poison_epoch * batches_per_epoch
    print(num_iterations)
    print(f'remain reg param : {remain_reg_param}')
    print(f'gamma: {gamma}')
    print(f'lamda: {lamda} ')
    print(f'use_logits: {use_logits}')

    for itr in tqdm.tqdm(range(num_iterations)):

        x, y = forget_data_gen.__next__()
        x_rem, y_rem = remain_data_gen.__next__()

        x = x.to(device)
        y = y.to(device)

        if len(y.shape)>1:
            y=y.squeeze()

        if y.dim() == 0:
            y = y.unsqueeze(0)


        x_rem = x_rem.to(device)
        y_rem = y_rem.to(device)

        # Generate adversarial examples
        test_model.eval()
        
        #inner loop         
        x_adv = adv.perturb(x, y, target_y=None, model=test_model, device=device, dist=dist, lamda = lamda, l1_norm =l1_norm,
                             current_iter=itr+1,total_iterations=num_iterations, scaling=scaling, gamma=gamma)
        

        adv_logits = test_model(x_adv)
        pred_label = torch.argmax(adv_logits, dim=1)

        # In the last epoch, store the predicted labels of the adversarial examples
        if itr >= (poison_epoch - 1) * batches_per_epoch:
            nearest_label.append(pred_label.tolist())
        
        # Update performance tracking
        num_hits += (y != pred_label).float().sum()        
        num_sum += y.shape[0]


        # Adversarial training step for the unlearn model
        unlearn_model.train()
        unlearn_model.zero_grad()
        optimizer.zero_grad()


        if remain_reg_param > 0:
            remain_logits = unlearn_model(x_rem)
            if use_logits == True:
                x_logit = test_model(x_rem)
                remain_loss = criterion(remain_logits, x_logit)
            else:
                remain_loss = criterion(remain_logits, y_rem)

            remain_contrib = remain_loss * remain_reg_param
        else:
            remain_contrib = 0

        
        ori_logits = unlearn_model(x)
        if use_logits == False:
            ori_loss = criterion(ori_logits, pred_label)
        else:
            if logit_preprocess:
                adv_logits = preprocess_logits(adv_logits, top_k=3)
            ori_loss = criterion(ori_logits, adv_logits)
 
        loss = ori_loss  + remain_contrib


        loss.backward()
        optimizer.step()

    print('attack success ratio:', (num_hits / num_sum).float())
    # print(nearest_label)
    print('boundary shrink time:', (time.time() - start_time))
    boundary_shrink_time = time.time() - start_time



    # np.save('nearest_label', nearest_label)


    torch.save(unlearn_model,output_name+'.pth')
    
    if not custom_forget:
        if selective_unlearning:
            test_forget_loader = train_forget_loader
            test_remain_loader = train_remain_loader
        else:
            test_forget_loader, test_remain_loader = get_forget_loader(dv, forget_class)
            _, train_remain_loader = get_forget_loader(dt, forget_class)

    elif custom_forget and not oculoplastics:
        test_forget_loader, test_remain_loader = get_custom_forget_loader(dv, test_metadata, to_forget)
        _, train_remain_loader = get_custom_forget_loader(dt, train_metadata, to_forget)

    elif custom_forget and oculoplastics:
        test_forget_loader, test_remain_loader = get_custom_forget_loader_oculoplastics(dv, test_metadata)
        _, train_remain_loader = get_custom_forget_loader_oculoplastics(dt, train_metadata)

            

    mode = 'pruned' if False else ''
    _, test_acc = eval(model=unlearn_model, data_loader=test_loader, mode=mode, print_perform=False, device=device,
                       name='test set all class')
    
    _, forget_acc = eval(model=unlearn_model, data_loader=test_forget_loader, mode=mode, print_perform=False,
                         device=device, name='test set forget class')
    
    _, remain_acc = eval(model=unlearn_model, data_loader=test_remain_loader, mode=mode, print_perform=False,
                         device=device, name='test set remain class')
    
    # unlearn_model.to('cpu')
    # ori_model.to(device)

    # _, forget_acc_orig = eval(model=ori_model, data_loader=test_forget_loader, mode=mode, print_perform=False,
    #                      device=device, name='test set forget class original model')
    
    # _, remain_acc_orig = eval(model=ori_model, data_loader=test_remain_loader, mode=mode, print_perform=False,
    #                      device=device, name='test set remain class original model')
    # ori_model.to('cpu')
    # retrain_model.to(device)

    # _, forget_acc_retrain = eval(model=retrain_model, data_loader=test_forget_loader, mode=mode, print_perform=False,
    #                      device=device, name='test set forget class retrain model')
    
    # _, remain_acc_retrain = eval(model=retrain_model, data_loader=test_remain_loader, mode=mode, print_perform=False,
    #                      device=device, name='test set remain class retrain model')
    # retrain_model.to('cpu')

    
    print('test acc:{:.2%}, forget acc:{:.2%}, remain acc:{:.2%}'
          .format(test_acc, forget_acc, remain_acc))
    
    # print('orig forget acc:{:.2%}, orig remain acc:{:.2%}, retrain forget acc:{:.2%}, retrain remain acc:{:.2%}'
    #       .format(forget_acc_orig, remain_acc_orig, forget_acc_retrain, remain_acc_retrain ))

    end = time.time()
    print('Time Consuming:', end - start, 'secs')

    unlearn_model.to(device)
    return unlearn_model, forget_acc, remain_acc, boundary_shrink_time




def preprocess_logits(logits, top_k=3):
    # Get the top K values and their indices
    topk_values, topk_indices = torch.topk(logits, top_k, dim=1)
    
    # Create a mask for the top K logits
    mask = torch.zeros_like(logits)
    mask.scatter_(1, topk_indices, topk_values)
    
    sparse_logits = mask
    
    sparse_logits_sum = sparse_logits.sum(dim=1, keepdim=True)
    sparse_logits = sparse_logits / sparse_logits_sum
    
    return sparse_logits

import argparse
import numpy as np
import boundary_unlearning
from utils import *
from trainer import *
import shutil
import os 
import pandas as pd
import time
import csv
import json
import torch
import torch.nn as nn
import torch.cuda
from params import get_parameters
from make_dataloaders import *
from embeddings import *
from ensemble import run_exps


def main(args):
    torch.cuda.empty_cache()
    seed_torch()
    
    # set up output file for dirty logging
    # TODO improve logging   
    # gamma_values = [.1, 0.5, 0.9]
    # gamma_values = [0, .0001, .1, 1]
    gamma_values = [0, .0001]

    # l1_norms = [True, False]
    l1_norms = [False]
    # distributions = ['normal', 'cauchy', 'laplacian', 'uniform']
    # distributions = ['normal', 'cauchy']
    distributions = ['normal']

    csv_columns, output_file_name = set_up_save(args, distributions, gamma_values, l1_norms, args.name)

    # set device 
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    create_dir(args.dataset_dir)
    create_dir(args.checkpoint_dir)

    path = args.checkpoint_dir + '/'
    model_name = path[2:] + args.model_name + args.data_name 

    combined_df = None

    if args.custom_unlearn:
        ord_df = pd.read_csv('csvs_fundus/Other_Retinal_Disorders_UNIQUE_MRN_filtered.csv')
        dr_df = pd.read_csv('csvs_fundus/Diabetic_Retinopathy_UNIQUE_MRN_filtered.csv')
        glauc_df = pd.read_csv('csvs_fundus/Glaucoma_UNIQUE_MRN_filtered.csv')
        combined_df = pd.concat([ord_df, dr_df, glauc_df], ignore_index=True)
    
    if args.custom_unlearn and args.oculoplastics:
        ted_df = pd.read_csv('csvs_oculoplastic/mm_07022024_full_run_TED_GT_pix.csv')
        cfd_df = pd.read_csv('csvs_oculoplastic/mm_07022024_full_run_CFD_GT_pix.csv')
        combined_df = pd.concat([ted_df, cfd_df], ignore_index=True)
        print(combined_df.head())
    

    trainset, testset, dataset = get_dataset(args.data_name, args.dataset_dir)
    train_loader, test_loader = get_dataloader(trainset, testset, args.batch_size, device=device)
    
    # set number of classes 
    num_classes, idx_to_class = set_num_classes(args, dataset)

    if args.data_name == 'oculoplastic':
        divide_by = 114
    elif args.data_name == 'fundus_3_class':
        divide_by = 1000
    elif args.data_name == 'oct_4_class':
        divide_by = 1500
    elif args.data_name == 'open_source':
        divide_by = 500
    elif args.data_name == 'dr_grade':
        divide_by = 62
    else:
        divide_by = 100000

    num_forget = int(divide_by * args.percent_to_forget)
    print(num_forget)

    train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
        train_forget_index, train_remain_index, test_forget_index, test_remain_index, train_dict, test_dict = dataloader_engine(args, trainset, testset, 
                                                                                                         combined_df, num_forget=num_forget, oculoplastics=args.oculoplastics)
    

    if args.tsne_embeddings:
        print('doing embeddings')

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        original_model = torch.load(args.original_model)
        retrained_model = torch.load(args.retrain_model)
        unlearned_model = torch.load(args.unlearn_model)

        models = [original_model, retrained_model, unlearned_model]
        titles = ['Original Model', 'Retrained Model', 'Unlearned Model']

        for i, (ax, model, title) in enumerate(zip(axes, models, titles)):
            model.to(device)
            
            embeddings, predictions, is_forget_sample = get_embeddings_predictions_and_forget_indications(
                model, test_forget_loader, test_remain_loader, device
            )
            
            plot_tsne(embeddings, predictions, is_forget_sample, title, ax, add_legend=(i == 0))
        
        plt.tight_layout()
        plt.savefig(args.embeddings_name + '.jpg', dpi=300)
        
        breakpoint



    if args.ensemble:
        print(len(train_remain_loader))

        run_exps(args, testset, trainset, train_remain_loader, finetune=True, frozen=False)



    ori_model, retrain_model, row_data = train_engine(args, train_remain_loader, test_remain_loader, train_loader, test_loader, 
                 dataset, num_classes, idx_to_class, device, model_name, output_file_name, 
                 csv_columns, distributions, gamma_values)

    if args.do_unlearning:
        if args.run_sota:
            print('DOING BOUNDARY SHRINKAGE WITH SOTA METHOD')
            save_me = f'SOTA_{args.data_name}_{args.forget_class}_{args.model_name}'

            unlearn_model_sota, forget_acc_sota, remain_acc_sota, unlearning_time = boundary_unlearning.boundary_shrink(
                    ori_model, train_forget_loader, trainset, testset, test_loader, device, 
                    forget_class=args.forget_class, path=path, custom_forget=args.custom_unlearn, to_forget=args.to_forget,
                    test_metadata=test_dict, train_metadata=train_dict,gamma=0, dist='normal',
                    output_name=save_me, use_linfpgd=args.use_linfpgd, lamda=0, l1_norm=False, data_name = args.data_name, scaling = None, 
                    oculoplastics=args.oculoplastics, retrain_model=retrain_model, train_remain_loader=train_remain_loader, use_logits = args.use_logits,
                    remain_reg_param= args.remain_reg, logit_preprocess= args.logit_preprocess
            )

            # Calculate per class accuracy
            per_class_accs_sota = test(unlearn_model_sota, test_loader, idx_to_class, num_classes, device)

            print(f'SOTA UNLEARNING TIME forgetting {num_forget} SAMPLES: {unlearning_time}')

            # Update the CSV file with the SOTA results
            with open(output_file_name, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                row_data['Forget Acc SOTA'] = forget_acc_sota.detach().item()
                row_data['Remain Acc SOTA'] = remain_acc_sota.detach().item()
                row_data['Unlearning Time'] = unlearning_time
                row_data['Per Class Accuracies SOTA'] = json.dumps(per_class_accs_sota)
                writer.writerow(row_data)

        # If using closest points regularization
        if args.closest_points and not args.specific_settings:
            print('DOING BOUNDARY SHRINKAGE WITH multiple GAMMAS AND DISTRIBUTIONS')
            for dist in distributions:
                for lamda in gamma_values:
                    for l1_setting in l1_norms:
                        save_me = f'CLOSEST_{args.data_name}_{args.forget_class}_{lamda}_{dist}_{args.scaling}'
                        unlearn_model, forget_acc, remain_acc, _ = boundary_unlearning.boundary_shrink(
                            ori_model, train_forget_loader, trainset, testset, test_loader, device,
                            forget_class=args.forget_class, path=path, custom_forget=args.custom_unlearn, to_forget=args.to_forget,
                            test_metadata=test_dict, train_metadata=train_dict, gamma=0,
                            dist=dist, output_name=save_me, use_linfpgd=args.use_linfpgd, lamda=lamda, l1_norm=l1_setting, data_name = args.data_name, scaling = args.scaling, 
                            oculoplastics=args.oculoplastics, retrain_model=retrain_model, train_remain_loader=train_remain_loader, use_logits = args.use_logits,
                        remain_reg_param= args.remain_reg, logit_preprocess= args.logit_preprocess
                        )
                        per_class_accs = test(unlearn_model, test_loader, idx_to_class, num_classes, device)

                        # Update row data
                        row_data[f'Forget Acc {dist}_lambda_{lamda}_{l1_setting}'] = forget_acc.detach().item()
                        row_data[f'Remain Acc {dist}_lambda_{lamda}_{l1_setting}'] = remain_acc.detach().item()
                        row_data[f'Per Class Accuracies {dist}_lambda_{lamda}_{l1_setting}'] = json.dumps(per_class_accs)

                        with open(output_file_name, 'a', newline='') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                            writer.writerow(row_data)

        # If sgld algorithm
        if args.sgld and not args.specific_settings:
            print('DOINT UNLEARNING WITH SGLD and closest points')
            for dist in distributions:
                for gamma in gamma_values:
                    save_me = f'SGLD_{args.data_name}_{args.forget_class}_gamma:_{gamma}_{dist}_lamda:{args.lamda}_remain_reg{args.remain_reg}'
                    unlearn_model, forget_acc, remain_acc, _ = boundary_unlearning.boundary_shrink(
                        ori_model, train_forget_loader, trainset, testset, test_loader, device,
                        forget_class=args.forget_class, path=path, custom_forget=args.custom_unlearn, to_forget=args.to_forget,
                        test_metadata=test_dict, train_metadata=train_dict, gamma=gamma,
                        dist=dist, output_name=save_me, use_linfpgd=args.use_linfpgd, lamda=args.lamda, l1_norm=False, data_name = args.data_name, scaling = None, 
                        oculoplastics=args.oculoplastics, retrain_model=retrain_model, train_remain_loader=train_remain_loader, use_logits = args.use_logits,
                        remain_reg_param= args.remain_reg, logit_preprocess= args.logit_preprocess
                    )
                    per_class_accs = test(unlearn_model, test_loader, idx_to_class, num_classes, device)

                    # Update row data
                    row_data[f'Forget Acc {dist} {gamma}'] = forget_acc.detach().item()
                    row_data[f'Remain Acc {dist} {gamma}'] = remain_acc.detach().item()
                    # row_data[f'Per Class Accuracies {dist} {gamma}'] = json.dumps(per_class_accs)

                    with open(output_file_name, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                        writer.writerow(row_data)

        # If specific settings
        if args.specific_settings:
            print('DOINT UNLEARNING WITH specific setting')

            # UNLEARNING SPECIFIC MODEL
            lamda = args.lamda
            gamma = args.gamma
            dist = args.dist
            print(gamma)
            save_me = f'SPECIFOC_{args.data_name}_{args.to_forget}_lamnda_{lamda}_gamma_{gamma}_{dist}_{args.model_name}'

            unlearn_model, forget_acc, remain_acc, boundary_shrink_time = boundary_unlearning.boundary_shrink(
                ori_model, train_forget_loader, trainset, testset, test_loader, device,
                forget_class=args.forget_class, path=path, custom_forget=args.custom_unlearn, to_forget=args.to_forget,
                test_metadata=test_dict, train_metadata=train_dict, gamma=gamma,
                dist=dist, output_name=save_me, use_linfpgd=False, lamda=lamda, l1_norm=False, data_name = args.data_name,scaling = None, 
                oculoplastics=args.oculoplastics, retrain_model=retrain_model
            )
            unlearn_model.to(device)

            print(f'MODIFIED UNLEARNING TIME for gamma {args.gamma} lambda {args.lamda} forgetting {num_forget} SAMPLES: {boundary_shrink_time} ')

            per_class_accs = test(unlearn_model, test_loader, idx_to_class, num_classes, device)

            # Update row data
            row_data[f'Forget Acc {dist}_lambda_{gamma}_{l1_setting}'] = forget_acc.detach().item()
            row_data[f'Remain Acc {dist}_lambda_{gamma}_{l1_setting}'] = remain_acc.detach().item()
            row_data[f'Per Class Accuracies {dist}_lambda_{gamma}_{l1_setting}'] = json.dumps(per_class_accs)

            with open(output_file_name, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writerow(row_data)



if __name__ == '__main__':
    args = get_parameters()
    main(args)





















import numpy as np
import torch
from scrub import scrub_met
from euk import cfk_unlearn,euk_unlearn
from neggrad import *
from finetune import finetune
from baseline_utils import *
import sys
sys.path.append('/home/unlearn-oph/deep_unlearning_2')
from models import *

from path_dicts import model_paths, selective_forget_models,chen_paths,ravi_paths,med_unlearn_paths
from tqdm import tqdm


def test(model, loader, idx_to_class, num_classes, device): 
    model.eval()
    correct = [0] * num_classes
    cnt = [0] * num_classes
    class_accuracies = {}
    # print('HERE')
    # print(idx_to_class)

    with torch.no_grad():
        for _, (data, target) in enumerate(tqdm(loader, leave=False)):
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            
            for i in range(target.size(0)):
                label = target[i].item()
                if pred[i].item() == label:
                    correct[label] += 1
                cnt[label] += 1

    for i in range(num_classes):
        accuracy = 0. if cnt[i] == 0 else correct[i] / cnt[i]
        class_name = idx_to_class[i]
        class_accuracies[class_name] = accuracy
    # print(class_accuracies)
    return class_accuracies



def all_readouts(model, test_loader, final_forget_loader, final_remain_loader, seed=2022, name='method'):
    _, test_acc = eval(model=model, data_loader=test_loader, device=device, name='test set all class')
    _, forget_acc = eval(model=model, data_loader=final_forget_loader, device=device, name='test set forget class')
    _, remain_acc = eval(model=model, data_loader=final_remain_loader, device=device, name='test set remain class')
    
    
    per_class_accs = test(model, test_loader, idx_to_class, num_classes, device)

    MIA = membership_inference_attack(model, test_loader, final_forget_loader, device, seed=seed)

    print(f"{name} -> Full test Acc: {test_acc:.5f} Forget Acc: {forget_acc:.5f} Remain Acc: {remain_acc:.5f} MIA: {np.mean(MIA):.2f}±{np.std(MIA):0.2f}")
    
    return dict(
        test_error=float(test_acc),
        forget_error=float(forget_acc),
        retain_error=float(remain_acc),
        MIA_mean=float(np.mean(MIA)),
        MIA_std=float(np.std(MIA)),
        per_class=per_class_accs
    )


if __name__ == '__main__':


    BASELINE_DIR = '/home/unlearn-oph/deep_unlearning_2/model_checkpoints/baseline_models'

    seed = 2022
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    retain_bs = 32
    forget_bs = 16
    batch_size = 8
    # readouts = {}
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    # unlearn_type = 'scrub'
    methods = ['finetune', 'neggrad', 'cfk', 'euk', 'scrub','ravi', 'chen','eval_orig']
    # methods = ['finetune', 'neggrad', 'cfk', 'euk', 'scrub','eval_orig']
    # methods = ['ravi', 'chen']

    # methods = [  ]
    # methods = ['eval_orig' ]

    SELECTIVE_UNLEARNING = False
    oculoplastics =  False
    med_unlearn = True
    combined_df = None

    # percentages = [.01, .1, .25, .5, .75]


    if SELECTIVE_UNLEARNING and not med_unlearn:
        model_paths = selective_forget_models
    if med_unlearn:
        model_paths = med_unlearn_paths
    
    for i, (data_name, model_list) in enumerate(model_paths.items()):
        if data_name == 'mri':
            percentages = [.1, .25, .5, .75]
        elif data_name == 'fashionmnist':
            percentages = [.01, .1, .25, .5, .75]
        else:
            percentages = [1]
        # try:
        for unlearn_type in methods:
            readouts = {}

            for percentage in percentages:

                print(f"Iteration {i}: {data_name} -> {model_list}")

                if SELECTIVE_UNLEARNING == False:
                    percentage = 1
                    if unlearn_type not in readouts:
                        readouts[unlearn_type] = {}
                    if data_name not in readouts[unlearn_type]:
                        readouts[unlearn_type][data_name] = {}

                    orig_model_path = model_list[0]
                    retrain_model_path = model_list[1]
                    model_type = model_list[2]

                    if med_unlearn == True:
                        forget_class = int(model_list[3])
                        custom_unlearn = model_list[4]


                else:
                    print('💀BEWARE: MAKE SURE YOU REALLY WANT TO DO THIS💀')
                    if unlearn_type not in readouts:
                        readouts[unlearn_type] = {}
                    if data_name not in readouts[unlearn_type]:
                        readouts[unlearn_type][data_name] = {}
                    if str(percentage) not in readouts[unlearn_type][data_name]:
                        readouts[unlearn_type][data_name][str(percentage)] = {}

                    orig_model_path = model_list['original']
                    retrain_model_path = model_list['retrain'][str(percentage)]
                    model_type = model_list['model_type']


                if data_name == 'open_source':
                    data_path = '/home/unlearn-oph/deep_unlearning_2/data/fundus_open_source'

                elif data_name == 'mri':
                    data_path = '/home/unlearn-oph/deep_unlearning_2/data/mri_unlearn' 

                elif data_name == 'ultrasound':
                    data_path = '/home/unlearn-oph/deep_unlearning_2/data/ultrasound_unlearn_oversample'

                elif data_name == 'oct_4_class':
                    data_path = '/home/unlearn-oph/deep_unlearning_2/data/oct_open_source'

                elif data_name == 'oculoplastic':
                    data_path = '/home/unlearn-oph/deep_unlearning_2/data/oculoplastic'
                    oculoplastics =True
                    if custom_unlearn:
                        ted_df = pd.read_csv('/home/unlearn-oph/deep_unlearning_2/data/csvs_oculoplastic/mm_07022024_full_run_TED_GT_pix.csv')
                        cfd_df = pd.read_csv('/home/unlearn-oph/deep_unlearning_2/data/csvs_oculoplastic/mm_07022024_full_run_CFD_GT_pix.csv')
                        combined_df = pd.concat([ted_df, cfd_df], ignore_index=True)
                        # print(combined_df.head())

                elif data_name == 'fundus_3_class':
                    data_path = '/home/unlearn-oph/deep_unlearning_2/data/fundus_big'
                    if custom_unlearn:
                        ord_df = pd.read_csv('/home/unlearn-oph/deep_unlearning_2/data/csvs_fundus/Other_Retinal_Disorders_UNIQUE_MRN_filtered.csv')
                        dr_df = pd.read_csv('/home/unlearn-oph/deep_unlearning_2/data/csvs_fundus/Diabetic_Retinopathy_UNIQUE_MRN_filtered.csv')
                        glauc_df = pd.read_csv('/home/unlearn-oph/deep_unlearning_2/data/csvs_fundus/Glaucoma_UNIQUE_MRN_filtered.csv')
                        combined_df = pd.concat([ord_df, dr_df, glauc_df], ignore_index=True)
        
                else:
                    data_path = './data'

                print(f'EXPERIMENTAL REPORT: \ncustom unlearn :  {custom_unlearn}, \n data name : {data_name} \n oculoplastics : {oculoplastics} \n forget class : {forget_class} \n unlearn type : {unlearn_type} ')
                
                trainset, testset, dataset = get_dataset(data_name, data_path)
                train_loader, test_loader = get_dataloader(trainset, testset, batch_size, device=device)

                # set number of classes 
                num_classes, idx_to_class = set_num_classes(data_name, dataset)
                total_forget_class = sum(1 for _, target in dataset if target == forget_class)
                num_forget = int(total_forget_class * percentage)
                print(f"Forget Percentage: {percentage}, Number to Forget: {num_forget}")

                # train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
                # train_forget_index, train_remain_index, test_forget_index, test_remain_index,train_dict, test_dict = dataloader_engine(batch_size, trainset, testset, num_forget=num_forget, selective_unlearning=SELECTIVE_UNLEARNING)
                
                train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
                train_forget_index, train_remain_index, test_forget_index, test_remain_index, train_dict, test_dict = dataloader_engine(batch_size, trainset, testset, 
                                                                                                                combined_df, num_forget=num_forget, forget_class = forget_class, 
                                                                                                                oculoplastics=oculoplastics, custom_unlearn = custom_unlearn,
                                                                                                                selective_unlearning = SELECTIVE_UNLEARNING)
            

                if SELECTIVE_UNLEARNING and not med_unlearn:
                    final_forget_loader = train_forget_loader
                    final_remain_loader = train_remain_loader
                    assert set(train_forget_index) == set(test_forget_index), "Train and test forget indices do not match!"
                    assert set(train_remain_index) == set(test_remain_index), "Train and test remain indices do not match!"
                else:
                    # final_forget_loader, final_remain_loader = get_forget_loader(testset, 0)
                    if not custom_unlearn:
                        if SELECTIVE_UNLEARNING:
                            final_forget_loader = train_forget_loader
                            final_remain_loader = train_remain_loader
                        else:
                            final_forget_loader, final_remain_loader = get_forget_loader(testset, forget_class)
                            # _, _ = get_forget_loader(trainset, forget_class)

                    elif custom_unlearn and not oculoplastics:
                        final_forget_loader, final_remain_loader = get_custom_forget_loader(testset, test_dict, 'Cirrus 800 FA')
                        # _, _ = get_custom_forget_loader(trainset, train_dict, 'Cirrus 800 FA')

                    elif custom_unlearn and oculoplastics:
                        final_forget_loader, final_remain_loader = get_custom_forget_loader_oculoplastics(testset, test_dict)
                        # _, _ = get_custom_forget_loader_oculoplastics(trainset, train_dict)

                model = load_model(model_type, num_classes=num_classes, data_name =data_name).to(device)
                model = load_model_state(model, orig_model_path)

                if unlearn_type == 'finetune':
                    print("Forgetting by Fine-tuning:")
                    ft_lr = 0.04
                    model_ft = model
                    ft_epochs = 10

                    finetune(model_ft, train_remain_loader, epochs=ft_epochs, quiet=True, lr=ft_lr)
                    if not SELECTIVE_UNLEARNING:
                        readouts[unlearn_type][data_name] = all_readouts(model_ft, test_loader, final_forget_loader, final_remain_loader, name='Finetune', seed=seed)
                    else:
                        readouts[unlearn_type][data_name][str(percentage)] = all_readouts(model_ft, test_loader, final_forget_loader, final_remain_loader, name='Finetune', seed=seed)

                elif unlearn_type == 'neggrad':
                    print("Forgetting by NegGrad:")
                    model_ng = model
                    ng_alpha = 0.9999
                    ng_epochs = 5
                    ng_lr = 0.01
                    negative_grad(model_ng, train_remain_loader, train_forget_loader, alpha=ng_alpha, epochs=ng_epochs, quiet=True, lr=ng_lr)
                    if not SELECTIVE_UNLEARNING:
                        readouts[unlearn_type][data_name] = all_readouts(model_ng, test_loader, final_forget_loader, final_remain_loader, name='NegGrad', seed=seed)
                    else:
                        readouts[unlearn_type][data_name][str(percentage)] = all_readouts(model_ng, test_loader, final_forget_loader, final_remain_loader, name='NegGrad', seed=seed)

                elif unlearn_type == 'cfk':
                    print("Forgetting by CFK:")
                    model_cfk = cfk_unlearn(model, train_remain_loader, model_type)
                    if not SELECTIVE_UNLEARNING:
                        readouts[unlearn_type][data_name] = all_readouts(model_cfk, test_loader, final_forget_loader, final_remain_loader, name='CFK', seed=seed)
                    else:
                        readouts[unlearn_type][data_name][str(percentage)] = all_readouts(model_cfk, test_loader, final_forget_loader, final_remain_loader, name='CFK', seed=seed)

                elif unlearn_type == 'euk':
                    print("Forgetting by EUK:")
                    model_initial = model
                    model_euk = euk_unlearn(model, train_remain_loader, model_type)
                    if not SELECTIVE_UNLEARNING:
                        readouts[unlearn_type][data_name] = all_readouts(model_euk, test_loader, final_forget_loader, final_remain_loader, name='EUK', seed=seed)
                    else:
                        readouts[unlearn_type][data_name][str(percentage)] = all_readouts(model_euk, test_loader, final_forget_loader, final_remain_loader, name='EUK', seed=seed)

                elif unlearn_type == 'scrub':
                    print("Forgetting by SCRUB:")
                    teacher = model
                    student = model

                    model_s, model_s_final = scrub_met(teacher, student, train_remain_loader, train_forget_loader, model_type, data_name)
                    if not SELECTIVE_UNLEARNING:
                        readouts[unlearn_type][data_name] = {
                            "SCRUB-R": all_readouts(model_s, test_loader, final_forget_loader, final_remain_loader, name='SCRUB-R', seed=seed),
                            "SCRUB": all_readouts(model_s_final, test_loader, final_forget_loader, final_remain_loader, name='SCRUB', seed=seed)
                        }
                    else:
                        readouts[unlearn_type][data_name][str(percentage)] = {
                            "SCRUB-R": all_readouts(model_s, test_loader, final_forget_loader, final_remain_loader, name='SCRUB-R', seed=seed),
                            "SCRUB": all_readouts(model_s_final, test_loader, final_forget_loader, final_remain_loader, name='SCRUB', seed=seed)
                        }

                elif unlearn_type == 'chen':
                    print("Forgetting by Chen et al.:")
                    model_chen = load_model(model_type, num_classes=num_classes, data_name=data_name).to(device)
                    if SELECTIVE_UNLEARNING:
                        chen_ckpt = model_list['chen'][str(percentage)]
                    else:
                        if med_unlearn and custom_unlearn:
                            chen_ckpt = os.path.join(BASELINE_DIR, chen_paths[data_name][1])
                        if med_unlearn and not custom_unlearn:
                            chen_ckpt = os.path.join(BASELINE_DIR, chen_paths[data_name][0])

                    print(f"Loading Chen baseline from {chen_ckpt}")
                    model_chen = load_checkpoint_without_dataparallel(chen_ckpt, model_chen)
                    if not SELECTIVE_UNLEARNING:
                        readouts[unlearn_type][data_name] = all_readouts(model_chen, test_loader, final_forget_loader, final_remain_loader, name='Chen', seed=seed)
                    else:
                        readouts[unlearn_type][data_name][str(percentage)] = all_readouts(model_chen, test_loader, final_forget_loader, final_remain_loader, name='Chen', seed=seed)

                elif unlearn_type == 'ravi':
                    print("Forgetting by Ravi et al.:")
                    model_ravi = load_model(model_type, num_classes=num_classes, data_name=data_name).to(device)
                    if SELECTIVE_UNLEARNING:
                        ravi_ckpt = model_list['ravi'][str(percentage)]
                    else:
                        if med_unlearn and custom_unlearn:
                            ravi_ckpt = os.path.join(BASELINE_DIR, chen_paths[data_name][1])
                        if med_unlearn and not custom_unlearn:
                            ravi_ckpt = os.path.join(BASELINE_DIR, chen_paths[data_name][0])
                        
                    print(f"Loading Ravi baseline from {ravi_ckpt}")
                    model_ravi = load_checkpoint_without_dataparallel(ravi_ckpt, model_ravi)
                    if not SELECTIVE_UNLEARNING:
                        readouts[unlearn_type][data_name] = all_readouts(model_ravi, test_loader, final_forget_loader, final_remain_loader, name='Ravi', seed=seed)
                    else:
                        readouts[unlearn_type][data_name][str(percentage)] = all_readouts(model_ravi, test_loader, final_forget_loader, final_remain_loader, name='Ravi', seed=seed)

                elif unlearn_type == 'eval_orig':
                    print("Evaluating Original and Retrain Models:")
                    model0 = load_model(model_type, num_classes=num_classes, data_name=data_name).to(device)
                    model0 = load_model_state(model0, retrain_model_path)

                    if not SELECTIVE_UNLEARNING:
                        readouts[unlearn_type][data_name] = {
                            # "Original": all_readouts(model, test_loader, test_forget_loader, test_remain_loader, name='Original', seed=seed),
                            "Retrain": all_readouts(model0, test_loader, final_forget_loader, final_remain_loader, name='Retrain', seed=seed)
                        }
                    else:
                        readouts[unlearn_type][data_name][str(percentage)] = {
                            # "Original": all_readouts(model, test_loader, test_forget_loader, test_remain_loader, name='Original', seed=seed),
                            "Retrain": all_readouts(model0, test_loader, final_forget_loader, final_remain_loader, name='Retrain', seed=seed)
                        }


            import json
            output_file = f"med_unlearn_{data_name}_{unlearn_type}_{custom_unlearn}_{forget_class}.json"  
            with open(output_file, "w") as f:
                json.dump(readouts, f, indent=4)
            custom_unlearn= False
            oculoplastics =  False

            print(f"Results saved to {output_file}")
        # except:
        #     pass

# train_loader_full, valid_loader_full, test_loader_full = datasets.get_loaders(dataset, batch_size=batch_size, seed=seed, root=dataroot, augment=False, shuffle=True)
# marked_loader, _, _ = datasets.get_loaders(dataset, class_to_replace= forget_class, num_indexes_to_replace=num_to_forget, only_mark=True, batch_size=1, seed=s, root=args.dataroot, augment=False, shuffle=True)

# forget_dataset = copy.deepcopy(marked_loader.dataset)
# marked = forget_dataset.targets < 0

# forget_dataset.data = forget_dataset.data[marked]
# forget_dataset.targets = - forget_dataset.targets[marked] - 1

# #forget_loader = torch.utils.data.DataLoader(forget_dataset, batch_size=args.forget_bs,num_workers=0,pin_memory=True,shuffle=True)
# forget_loader = replace_loader_dataset(train_loader_full, forget_dataset, batch_size=forget_bs, seed=seed, shuffle=True)

# retain_dataset = copy.deepcopy(marked_loader.dataset)
# marked = retain_dataset.targets >= 0
# retain_dataset.data = retain_dataset.data[marked]
# retain_dataset.targets = retain_dataset.targets[marked]

# #retain_loader = torch.utils.data.DataLoader(retain_dataset, batch_size=args.retain_bs,num_workers=0,pin_memory=True,shuffle=True)
# retain_loader = replace_loader_dataset(train_loader_full, retain_dataset, batch_size=retain_bs, seed=seed, shuffle=True)

# assert(len(forget_dataset) + len(retain_dataset) == len(train_loader_full.dataset))



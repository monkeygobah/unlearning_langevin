import time
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch
from torch import nn, optim
from tqdm import tqdm
from models import AllCNN, CustomResNet, MNISTNet
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
# from linformer import Linformer
# from vit_pytorch.efficient import ViT
import csv
import timm 
from torchvision import datasets
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def loss_picker(loss, train_loader=None, device='cpu'):
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'cross':
        if train_loader is not None:
            # Extract labels from the train_loader
            train_labels = []
            for _, labels in train_loader:
                train_labels.extend(labels.numpy())  

            # Compute class weights
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

            # Define the CrossEntropyLoss with class weights
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            class_weights.to('cpu')
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        print("Automatically assigning MSE loss function to you...")
        criterion = nn.MSELoss()

    return criterion

def optimizer_picker(optimization, param, lr, momentum=0.):
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        print('Using SGD for optimization')
        optimizer = optim.SGD(param, lr=lr, momentum=momentum, weight_decay=1e-4)
    else:
        print("NOTHING SELECTED FOR OPTIMZER")
        
    return optimizer

def train(model, data_loader, criterion, optimizer, loss_mode, device='cpu'):
    running_loss = 0
    model.train()
    print(len(data_loader))

    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if len(batch_y.shape) > 1:
            batch_y = batch_y.squeeze()  

        optimizer.zero_grad()
        
        output = model(batch_x) 


        if loss_mode == "mse":
            loss = criterion(output, batch_y)  
        elif loss_mode == "cross":
            loss = criterion(output, batch_y)  
        elif loss_mode == 'neg_grad':
            loss = -criterion(output, batch_y)
 
        loss.backward()
        optimizer.step()
        running_loss += loss
    return running_loss


def train_save_model(train_loader, test_loader, model_name, optim_name, learning_rate, num_epochs, device, path, dataset=None, relearning=False, unlearned_model=None, data_name=None):
    start = time.time()
    losses = []
    accuracies = []
    
    if dataset:
        if isinstance(dataset, datasets.SVHN):
            original_targets = dataset.labels  
        elif isinstance(dataset, (datasets.MNIST, datasets.CIFAR10)):
            if isinstance(dataset.targets, torch.Tensor):
                original_targets = dataset.targets.tolist()
            else:
                original_targets = dataset.targets
        else:
            original_targets = [dataset.imgs[i][1] for i in range(len(dataset))]

        if data_name == 'medmnist':
            num_classes = 9
        else:
            num_classes = len(set(original_targets))
            print(num_classes)
    else:
        num_classes = max(train_loader.dataset.targets) + 1


    if model_name == 'resnet':
        model = CustomResNet(num_classes=num_classes)
        model = nn.DataParallel(model) 
        model.to(device)
        
    elif model_name == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, img_size=512, patch_size=32)
        model.head = nn.Linear(model.head.in_features, num_classes)  # Adjust the final layer
        model.to(device)

    elif model_name == 'MNISTNet':
        model = MNISTNet()
        model.to(device)
    
    elif model_name == 'AllCNN':
        if data_name == 'fashionmnist':
            model = AllCNN(n_channels=1, num_classes=num_classes) 
        elif data_name =='medmnist':
            model = AllCNN(n_channels=3, num_classes=num_classes) 
        elif data_name =='svhn':
            model = AllCNN(n_channels=3, num_classes=num_classes)  
        elif data_name =='cifar10':
            model = AllCNN(n_channels=3, num_classes=num_classes)  

        model = nn.DataParallel(model) 
        model.to(device)


    criterion = loss_picker('cross')
    optimizer = optimizer_picker(optim_name, model.parameters(), lr=learning_rate, momentum=0.9)

    best_acc = 0
    SAVE = 30
    
    for epo in range(num_epochs):
        print('EPOCH:{}'.format(epo+1))
        loss = train(model=model, data_loader=train_loader, criterion=criterion, optimizer=optimizer, loss_mode='cross',
              device=device)
        
        print(f'training loss is: {loss}')
        losses.append(loss.item() / len(train_loader))


        _, acc = eval(model=model, data_loader=test_loader, mode='', print_perform=False, device=device)
        accuracies.append(acc.item())  
        print('test acc:{}'.format(acc))

        if acc>=best_acc:
            best_acc = acc

        # if (epo+1) % SAVE == 0:
            print('SAVING')
            print(f'current acc = {acc}')
            print(f'best acc = {best_acc}')
            torch.save(model, f'{path}{epo+1}.pth')

  

        if (epo+1) == num_epochs:
            print('SAVING LAST EPOCH')
            torch.save(model, f'{path}{epo+1}_final_model_{acc}.pth')


    end = time.time()
    print('training time:', end-start, 's')
    
    csv_path = f'{path}training_metrics.csv'
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss', 'Accuracy'])
        for epoch, (loss, acc) in enumerate(zip(losses, accuracies), 1):
            writer.writerow([epoch, loss, acc])    
            
    #     # Plotting
    # fig, ax1 = plt.subplots()

    # color = 'tab:red'
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Loss', color=color)
    # ax1.plot(range(num_epochs), losses, color=color)
    # ax1.tick_params(axis='y', labelcolor=color)

    # ax2 = ax1.twinx()  
    # color = 'tab:blue'
    # ax2.set_ylabel('Accuracy', color=color)  
    # ax2.plot(range(num_epochs), accuracies, color=color)
    # ax2.tick_params(axis='y', labelcolor=color)

    # fig.tight_layout()  #
    # plt.savefig(f'{path}training_metrics.jpg')
    
    return model, num_classes, end


def test(model, loader, idx_to_class, num_classes, device): 
    model.eval()
    correct = [0] * num_classes
    cnt = [0] * num_classes
    class_accuracies = {}


    with torch.no_grad():
        for _, (data, target) in enumerate(tqdm(loader, leave=False)):
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) 
            
            for i in range(target.size(0)):
                label = target[i].item()
                if pred[i].item() == label:
                    correct[label] += 1
                cnt[label] += 1

    for i in range(num_classes):
        accuracy = 0. if cnt[i] == 0 else correct[i] / cnt[i]
        class_name = idx_to_class[i]
        class_accuracies[class_name] = accuracy

    return class_accuracies


def eval(model, data_loader, batch_size=64, mode='backdoor', print_perform=False, device='cpu', name=''):
    model.eval() 
    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):
        
        if len(batch_y.shape)>1:
            batch_y=batch_y.squeeze()

        if batch_y.dim() == 0:
            batch_y = batch_y.unsqueeze(0)

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_y_predict = model(batch_x)

        batch_y_predict = torch.argmax(batch_y_predict, dim=1)

        y_predict.append(batch_y_predict)
        y_true.append(batch_y)

    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)

    num_hits = (y_true == y_predict).float().sum()
    acc = num_hits / y_true.shape[0]
    
 
    return accuracy_score(y_true.cpu(), y_predict.cpu()), acc


def train_engine(args, train_remain_loader, test_remain_loader, train_loader, test_loader, \
                 dataset, num_classes, idx_to_class, device, model_name, output_file_name, csv_columns, distributions, gamma_values,exp_name='deafault'):
    
    if args.train:
        print('=' * 100)
        print(' ' * 25 + 'train original model and retrain model from scratch')
        print('=' * 100)
        ori_model, num_classes, _ = train_save_model(train_loader, test_loader, args.model_name, args.optim_name, args.lr,
                                     args.epoch, device, model_name + "_original_model_", dataset=dataset, data_name=args.data_name)
        
        # if torch.cuda.device_count() > 1:
        #     ori_model = torch.nn.DataParallel(ori_model, device_ids =[args.gpu_id, 0,2])
            
        print('\noriginal model acc:\n', test(ori_model, test_loader, idx_to_class, num_classes, device))

        retrain_model, _, _ = train_save_model(train_remain_loader, test_remain_loader, args.model_name, args.optim_name,
                                        args.lr, args.epoch, device, model_name + "_retrain_model_" + 'class_' + str(args.forget_class) + '_', dataset=dataset, data_name=args.data_name)
        
        print('\nretrain model acc:\n', test(retrain_model, test_loader, idx_to_class, num_classes, device))    
        return ori_model, retrain_model, None
    
    elif args.retrain_only:
        ori_model = torch.load(args.original_model, map_location=torch.device('cpu')).to(device)
        ori_model.to('cpu')
        print(model_name + "_retrain_" + exp_name + '_' )
        retrain_model, _, time_retrain = train_save_model(train_remain_loader, test_remain_loader, args.model_name, args.optim_name,
                                        args.lr, args.epoch, device,  model_name + "_retrain_" + exp_name + '_' , dataset=dataset, data_name=args.data_name)

        # print('\nretrain model acc:\n', test(retrain_model, test_loader, idx_to_class, num_classes, device))   

        print(f'RETRAIN TIME {time_retrain}')

        return ori_model, retrain_model, None
 
    else:
        print('=' * 100)
        print(' ' * 25 + 'load original model and retrain model')
        print('=' * 100)

        # Load and print original model accuracy
        ori_model = torch.load(args.original_model, map_location=torch.device('cpu'))
        ori_model.to(device)

        _, orig_acc = eval(model=ori_model, data_loader=test_loader, mode='', print_perform=False, device=device)

        print('test acc:{}'.format(orig_acc))
        print('\n ORIGINAL model acc:\n', test(ori_model, test_loader, idx_to_class, num_classes, device))

        ori_model.to('cpu')


        retrain_model = torch.load(args.retrain_model, map_location=torch.device('cpu'))
        retrain_model.to(device)
        _, retrain_acc = eval(model=retrain_model, data_loader=test_remain_loader, mode='', print_perform=False, device=device)
        
        print('test acc retrain:{}'.format(retrain_acc))
        print('\nretrain model acc:\n', test(retrain_model, test_loader, idx_to_class, num_classes, device))

        retrain_model.to('cpu')

        # Log average accuracy of original and retrain models
        with open(output_file_name, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            row_data = {
                'Dataset': args.data_name,
                'Model': args.model_name,
                'Gamma': 'N/A',
                'Distribution': 'N/A',
                'Forget Acc SOTA': 'N/A',
                'Remain Acc SOTA': 'N/A',
                'Original Acc': orig_acc.detach().item(),
                'Retrain Acc': retrain_acc.detach().item(),
                'Unlearning Time': 'N/A',
                'Per Class Accuracies SOTA': 'N/A'
            }
            
            # Add entries for each combination of distribution and gamma value
            for dist in distributions:
                for gamma in gamma_values:
                    forget_acc_col = f'Forget Acc {dist} {gamma}'
                    remain_acc_col = f'Remain Acc {dist} {gamma}'
                    per_class_acc_col = f'Per Class Accuracies {dist} {gamma}'
                    row_data[forget_acc_col] = 'N/A'
                    row_data[remain_acc_col] = 'N/A'
                    row_data[per_class_acc_col] = 'N/A'
            
            writer.writerow(row_data)




        with open(output_file_name, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            row_data = {
                'Dataset': args.data_name,
                'Original Acc': orig_acc.detach().item(),
                'Retrain Acc': retrain_acc,

                'Retrain Acc': retrain_acc.detach().item(),
                'Unlearning Time': 'N/A',
            }
            
            for dist in distributions:
                for gamma in gamma_values:
                    forget_acc_col = f'Forget Acc {dist} {gamma}'
                    remain_acc_col = f'Remain Acc {dist} {gamma}'
                    row_data[forget_acc_col] = 'N/A'
                    row_data[remain_acc_col] = 'N/A'
                    row_data[per_class_acc_col] = 'N/A'
            
            writer.writerow(row_data)



        return ori_model, retrain_model, row_data



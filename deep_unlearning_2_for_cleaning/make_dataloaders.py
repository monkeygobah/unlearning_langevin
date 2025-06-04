import os
import random
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from datetime import datetime
from utils import *
from medmnist import INFO, Evaluator
import medmnist
from torchvision import datasets, transforms

def resize_width_pad_height(target_width=512, target_height=512):
    def transform(image):
        # Calculate the new height maintaining the aspect ratio
        aspect_ratio = image.width / image.height
        new_height = int(round(target_width / aspect_ratio))

        # Resize the image to have the correct width
        resize_transform = transforms.Resize((new_height, target_width))
        resized_image = resize_transform(image)

        # Calculate padding to add to the top and bottom to reach the target height
        padding_top = (target_height - new_height) // 2
        padding_bottom = target_height - new_height - padding_top

        # Pad the resized image to have the correct height
        pad_transform = transforms.Pad((0, padding_top, 0, padding_bottom), fill=0, padding_mode='constant')
        padded_image = pad_transform(resized_image)

        return padded_image

    return transform

def get_dataset(data_name, path='./data'):
    if data_name == 'dr_grade':
        img_size = 36
    else:
        img_size = 512

    if data_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST-specific normalization
        ])
        trainset = datasets.MNIST(root=path, train=True, download=True, transform=transform)
        testset = datasets.MNIST(root=path, train=False, download=True, transform=transform)
        dataset = datasets.MNIST(root=path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        return trainset, testset, dataset

    elif data_name == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 normalization
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        trainset = datasets.CIFAR10(root=path, train=True, download=True, transform=train_transform)
        testset = datasets.CIFAR10(root=path, train=False, download=True, transform=test_transform)
        dataset = datasets.CIFAR10(root=path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        return trainset, testset, dataset

    elif data_name == 'svhn':
        # SVHN has 3 channels and 32x32 images
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))  # SVHN-specific normalization
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))  # SVHN normalization
        ])
        
        trainset = datasets.SVHN(root=path, split='train', download=True, transform=train_transform)
        testset = datasets.SVHN(root=path, split='test', download=True, transform=test_transform)
        dataset = datasets.SVHN(root=path, split='test', download=True, transform=test_transform)
        return trainset, testset, dataset

    elif data_name == 'fashionmnist':
        # FashionMNIST is grayscale (1 channel) and has 28x28 images, so we resize to 32x32
        train_transform = transforms.Compose([
            transforms.Resize(32),  # Resize to 32x32
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # FashionMNIST-specific grayscale normalization
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(32),  # Resize to 32x32
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Grayscale normalization
        ])
        
        trainset = datasets.FashionMNIST(root=path, train=True, download=True, transform=train_transform)
        testset = datasets.FashionMNIST(root=path, train=False, download=True, transform=test_transform)
        dataset = datasets.FashionMNIST(root=path, train=False, download=True, transform=test_transform)
        return trainset, testset, dataset

    elif data_name == 'medmnist':
        info = INFO['pathmnist']  # You can adjust 'pathmnist' for different MedMNIST datasets

        n_channels = info['n_channels']
        n_classes = len(info['label'])
        print(n_classes, n_channels)
        DataClass = getattr(medmnist, info['python_class'])
        
        train_transform = transforms.Compose([
            transforms.Resize(32),  # Resize to 32x32
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # RGB normalization (assuming 3 channels)
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(32),  # Resize to 32x32
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # RGB normalization
        ])
        
        trainset = DataClass(split='train', transform=train_transform, download=True)
        testset = DataClass(split='test', transform=test_transform, download=True)
        dataset = DataClass(split='train', transform=transforms.Compose([transforms.ToTensor()]))
        return trainset, testset, dataset
            

    elif data_name == 'oculoplastic':
            print('IN OCULOPLASTIC')
            print(path)
            train_transforms = transforms.Compose([
                resize_width_pad_height(),
                transforms.RandomHorizontalFlip(), 
                transforms.RandomRotation(15), 
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
                transforms.RandomResizedCrop(512), 
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            test_transforms = transforms.Compose([
                resize_width_pad_height(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            dataset = datasets.ImageFolder(path)
            num_train = len(dataset)
            indices = list(range(num_train))
            
            split = int(np.floor(0.2 * num_train))
            np.random.shuffle(indices)
            
            train_idx, test_idx = indices[split:], indices[:split]

            train_dataset = datasets.ImageFolder(path, transform=train_transforms)
            test_dataset = datasets.ImageFolder(path, transform=test_transforms)

            trainset = torch.utils.data.Subset(train_dataset, train_idx)
            testset = torch.utils.data.Subset(test_dataset, test_idx)
            
            return trainset, testset, dataset

    else:
        train_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(15), 
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
            transforms.RandomResizedCrop(512), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        dataset = datasets.ImageFolder(path)
        num_train = len(dataset)
        indices = list(range(num_train))
        
        split = int(np.floor(0.2 * num_train))
        np.random.shuffle(indices)
        
        train_idx, test_idx = indices[split:], indices[:split]

        train_dataset = datasets.ImageFolder(path, transform=train_transforms)
        test_dataset = datasets.ImageFolder(path, transform=test_transforms)

        trainset = torch.utils.data.Subset(train_dataset, train_idx)
        testset = torch.utils.data.Subset(test_dataset, test_idx)
        
        return trainset, testset, dataset

def get_dataloader(trainset, testset, batch_size, device):
        
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def split_class_data(dataset, forget_class, num_forget):
    forget_index = []
    class_remain_index = []
    remain_index = []
    sum = 0

    for i, (data, target) in enumerate(dataset):
        if target == forget_class and sum < num_forget:
            forget_index.append(i)

            sum += 1
        elif target == forget_class and sum >= num_forget:
            class_remain_index.append(i)
            remain_index.append(i)
            sum += 1

        else:
            remain_index.append(i)
            

    return forget_index, remain_index, class_remain_index


def split_metadata_data(subset, metadata_dict, unlearn_attribute, num_forget):
    attributes = ['OS', 'OD', 'Spectralis (Scans)', 'Cirrus 800 FA', '2015', '2016', '2017', '2018']
    attr_index = attributes.index(unlearn_attribute)
    
    forget_index = []
    class_remain_index = []
    remain_index = []
    sum = 0
    
    original_dataset = subset.dataset  # Access the original dataset
    for i, subset_index in enumerate(subset.indices):  # Iterate through indices of the subset
        img_path, _ = original_dataset.imgs[subset_index]  # Access the image path using the original dataset
        filename = os.path.basename(img_path)
        if filename in metadata_dict:
            ohe_vector = metadata_dict[filename]
            if ohe_vector[attr_index] == 1 and sum < num_forget:
                forget_index.append(i)  # Use subset_index
                sum += 1
            elif ohe_vector[attr_index] == 1 and sum >= num_forget:
                class_remain_index.append(i)  # Use subset_index
                remain_index.append(i)  # Use subset_index
                sum += 1
            else:
                remain_index.append(i)  # Use subset_index

    return forget_index, remain_index, class_remain_index


def get_custom_unlearn_loader(trainset, testset, train_dict, test_dict, unlearn_attribute, batch_size):
    num_forget = 1000
    repair_num_ratio = 0.01  
    
    # Use the split_metadata_data function for both train and test sets
    train_forget_index, train_remain_index, class_remain_index = split_metadata_data(
        trainset, train_dict, unlearn_attribute, num_forget)
    
    test_forget_index, test_remain_index, _ = split_metadata_data(
        testset, test_dict, unlearn_attribute, num_forget=len(testset.dataset.imgs))

    # Sampling a subset of the class_remain_index for repairs
    repair_class_index = random.sample(class_remain_index, int(repair_num_ratio * len(class_remain_index)))

    # Creating samplers based on the indices
    train_forget_sampler = SubsetRandomSampler(train_forget_index)
    train_remain_sampler = SubsetRandomSampler(train_remain_index)
    
    repair_class_sampler = SubsetRandomSampler(repair_class_index)
    
    test_forget_sampler = SubsetRandomSampler(test_forget_index)
    test_remain_sampler = SubsetRandomSampler(test_remain_index)

    # Creating data loaders using the samplers
    train_forget_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, sampler=train_forget_sampler)
    train_remain_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, sampler=train_remain_sampler)

    repair_class_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, sampler=repair_class_sampler)
    
    test_forget_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, sampler=test_forget_sampler)
    test_remain_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, sampler=test_remain_sampler)

    return train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
           train_forget_index, train_remain_index, test_forget_index, test_remain_index


def get_unlearn_loader(trainset, testset, forget_class, batch_size, num_forget, repair_num_ratio=0.01, selective_unlearning = False):
   
    train_forget_index, train_remain_index, class_remain_index = split_class_data(trainset, forget_class,
                                                                                  num_forget=num_forget)
    
    if not selective_unlearning:
        test_forget_index, test_remain_index, _ = split_class_data(testset, forget_class, num_forget=len(testset))
    else:
        test_forget_index = train_forget_index
        test_remain_index = train_remain_index  


    repair_class_index = random.sample(class_remain_index, int(repair_num_ratio * len(class_remain_index)))

    train_forget_sampler = SubsetRandomSampler(train_forget_index)  # 5000
    train_remain_sampler = SubsetRandomSampler(train_remain_index)  # 45000

    repair_class_sampler = SubsetRandomSampler(repair_class_index)

    test_forget_sampler = SubsetRandomSampler(test_forget_index)  # 1000
    test_remain_sampler = SubsetRandomSampler(test_remain_index)  # 9000

    train_forget_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=train_forget_sampler)
    train_remain_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=train_remain_sampler)

    repair_class_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=repair_class_sampler)

    test_forget_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                                     sampler=test_forget_sampler)
    test_remain_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                                     sampler=test_remain_sampler)

    return train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
           train_forget_index, train_remain_index, test_forget_index, test_remain_index


def get_relearn_loader_from_remain(train_remain_loader, batch_size, metadata_df, exclude_feature='Cirrus 800 FA'):
    class_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Initialize lists for indices
    relearn_indices = []

    # Access the original dataset from the train_remain_loader
    original_dataset = train_remain_loader.dataset.dataset

    for idx in train_remain_loader.dataset.indices:
        img_path, target = original_dataset.imgs[idx]
        filename = os.path.basename(img_path)
        if target == 0:  # DR class
            relearn_indices.append(idx)
        elif target == 1 and filename in metadata_df['de_FileName'].values:
            if metadata_df[metadata_df['de_FileName'] == filename]['DeviceProc'].values[0] != exclude_feature:
                relearn_indices.append(idx)

    # Create a dataset with the filtered samples
    relearn_dataset = torch.utils.data.Subset(original_dataset, relearn_indices)
    relearn_dataset.dataset.transform = class_transforms

    # Create a DataLoader
    relearn_loader = DataLoader(relearn_dataset, batch_size=batch_size, shuffle=True)

    return relearn_loader


def get_forget_loader(dt, forget_class):
    idx = []
    els_idx = []
    count = 0
    for i in range(len(dt)):
        _, lbl = dt[i]
        if lbl == forget_class:
            # if forget:
            #     count += 1
            #     if count > forget_num:
            #         continue
            idx.append(i)
        else:
            els_idx.append(i)
    forget_loader = torch.utils.data.DataLoader(dt, batch_size=8, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(idx), drop_last=True)
    remain_loader = torch.utils.data.DataLoader(dt, batch_size=8, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(els_idx), drop_last=True)
    return forget_loader, remain_loader


def get_custom_forget_loader(dataset, metadata_dict, attribute_to_forget, batch_size=8):
    forget_indices = []
    remain_indices = []

    attribute_index = {
        'OS': 0, 'OD': 1, 'Spectralis (Scans)': 2, 'Cirrus 800 FA': 3,
        '2015': 4, '2016': 5, '2017': 6, '2018': 7
    }.get(attribute_to_forget)
    
    original_dataset = dataset.dataset
    
    
    if attribute_index is None:
        raise ValueError(f"Attribute {attribute_to_forget} not recognized.")

    for i, subset_index in enumerate(dataset.indices):
        img_path, _ = original_dataset.imgs[subset_index]  
        filename = os.path.basename(img_path)

        if filename in metadata_dict:
            ohe_vector = metadata_dict[filename]
            if ohe_vector[attribute_index] == 1:
                forget_indices.append(i)
            else:
                remain_indices.append(i)
      

    forget_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(forget_indices), drop_last=True)
    remain_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(remain_indices), drop_last=True)

    return forget_loader, remain_loader


def get_custom_forget_loader_oculoplastics(dataset, metadata_dict, batch_size=8):
    forget_indices = []
    remain_indices = []

    original_dataset = dataset.dataset

    for i, subset_index in enumerate(dataset.indices):
        img_path, _ = original_dataset.imgs[subset_index]
        filename = os.path.basename(img_path)

        if filename in metadata_dict:
            forget_indices.append(i)
        else:
            remain_indices.append(i)

    print(f'length of TEST forget indices :{len(forget_indices)}')
    print(f'length of TEST remain indices :{len(remain_indices)}')

    forget_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(forget_indices), drop_last=True)
    remain_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(remain_indices), drop_last=True)

    print(f'length of TEST forget loader :{len(forget_loader)}')
    print(f'length of TEST remain loader :{len(remain_loader)}')

    return forget_loader, remain_loader



def dataloader_engine(args, trainset, testset, combined_df, num_forget=5000, oculoplastics = False, selective_unlearning = False):
    if args.custom_unlearn and not oculoplastics:
        print('Getting CUSTOM Unlearn Loader using OHE of Metadata')
        train_dict = map_metadata(trainset, combined_df)
        test_dict = map_metadata(testset, combined_df)
        
        train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
        train_forget_index, train_remain_index, test_forget_index, test_remain_index \
            = get_custom_unlearn_loader(trainset, testset, train_dict, test_dict, args.to_forget, args.batch_size)
        
    elif args.custom_unlearn and oculoplastics:
        print('Getting CUSTOM Unlearn Loader using OHE of Metadata')
        train_dict = map_metadata_oculoplastics(trainset, combined_df)
        test_dict = map_metadata_oculoplastics(testset, combined_df)
        
        train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
        train_forget_index, train_remain_index, test_forget_index, test_remain_index \
            = get_custom_unlearn_loader_oculoplastics(trainset, testset, train_dict, test_dict, args.batch_size)

    
    else:
        print('getting unlearn loader')
        forget_class = args.forget_class
        num_forget = num_forget
        train_dict = None
        test_dict = None
        train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
        train_forget_index, train_remain_index, test_forget_index, test_remain_index \
            = get_unlearn_loader(trainset, testset, forget_class, args.batch_size, num_forget, selective_unlearning=selective_unlearning)
        
    return train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
        train_forget_index, train_remain_index, test_forget_index, test_remain_index, train_dict, test_dict


def map_metadata_oculoplastics(dataset, df, feature='vert_pf', threshold=11):
    metadata_dict = {}
    for img_path, _ in dataset.dataset.imgs:
        filename = os.path.basename(img_path)
        row = df[df['file'] == filename[:-9]]
        # print(row)
        if not row.empty:
            left_feature = row[f'left_{feature}'].values[0]
            right_feature = row[f'right_{feature}'].values[0]
            avg_feature = (left_feature + right_feature) / 2
            if avg_feature > threshold:
                metadata_dict[filename] = avg_feature
    print(f"Total images mapped with {feature} > {threshold}: {len(metadata_dict)}")
    return metadata_dict



def split_metadata_data_oculoplastics(subset, metadata_dict, num_forget):
    forget_index = []
    remain_index = []
    sum = 0

    original_dataset = subset.dataset
    for i, subset_index in enumerate(subset.indices):
        img_path, _ = original_dataset.imgs[subset_index]
        filename = os.path.basename(img_path)
        if filename in metadata_dict:
            if sum < num_forget:
                forget_index.append(i)
                sum += 1
            else:
                remain_index.append(i)
        else:
            remain_index.append(i)

    print(f"Total images to forget: {len(forget_index)}")
    print(f"Total images to remain: {len(remain_index)}")
    return forget_index, remain_index

def get_custom_unlearn_loader_oculoplastics(trainset, testset, train_dict, test_dict, batch_size, num_forget=1000, repair_num_ratio=0.01):
    train_forget_index, train_remain_index = split_metadata_data_oculoplastics(trainset, train_dict, num_forget)
    test_forget_index, test_remain_index = split_metadata_data_oculoplastics(testset, test_dict, num_forget=len(testset.dataset.imgs))

    repair_class_index = random.sample(train_remain_index, int(repair_num_ratio * len(train_remain_index)))

    train_forget_sampler = SubsetRandomSampler(train_forget_index)
    train_remain_sampler = SubsetRandomSampler(train_remain_index)
    repair_class_sampler = SubsetRandomSampler(repair_class_index)
    test_forget_sampler = SubsetRandomSampler(test_forget_index)
    test_remain_sampler = SubsetRandomSampler(test_remain_index)

    train_forget_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, sampler=train_forget_sampler)
    train_remain_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, sampler=train_remain_sampler)
    repair_class_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, sampler=repair_class_sampler)
    test_forget_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, sampler=test_forget_sampler)
    test_remain_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, sampler=test_remain_sampler)

    print(f"Train forget loader size: {len(train_forget_loader)}")
    print(f"Train remain loader size: {len(train_remain_loader)}")
    print(f"Repair class loader size: {len(repair_class_loader)}")
    print(f"Test forget loader size: {len(test_forget_loader)}")
    print(f"Test remain loader size: {len(test_remain_loader)}")

    return train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
           train_forget_index, train_remain_index, test_forget_index, test_remain_index
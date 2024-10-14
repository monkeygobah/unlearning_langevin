# Unlearning Using Langevin Dynamics
Deep Unlearning using Langevin Dynamics

## Overview
 Deep unlearning is a method aimed at intentionally forgetting specific information within a neural network without impacting its overall performance. This technique is particularly useful in scenarios where privacy concerns or data regulations require the removal of specific data from trained models Our novel formualtion includes tunable control over behavior of unlearning algorithm + ablation experiments, validation on 5 datsets (2 medical, 3 standard benchmark), SOTA performance on all datasets evaluated, and a novel method of composition of unlearned algorithms

![fig_1_aistats](https://github.com/user-attachments/assets/241c00e0-c4dc-49a7-900e-556c198def0b)


### Replicating experiments

If you are not using a provided pretrained checkpoint, you will need to use the ``` --train ``` flag the first time you run an experiment. ```---retrain``` allows you to retrain without the forget set present. For instance, in the following you would toggle train and retrain for CIFAR-10
```sh
python main.py --forget_class 0 \
                --data_name fashionmnist \
                --model_name AllCNN \
                --lr 0.01 \
                --epoch 15  \
                --batch_size 64  \
                --gpu_id 0 \
                --train \
```


The `do_unlearning` flag sets the algorithm to unlearn mode. SGLD instructs the algorithm to run with $\gamma \in [0,1e-4]$ and you can specify the lambda. The `use_sota` flag will run experiments using $\gamma=\lambda=0$. The `remain_reg` flag is how $\phi$ is defined, and the optional flag `use_logits` computes the loss on $F$ and $R$ using logits in Phase 2. Results will be saved in the file named via the `name` tag.

```sh
python main.py --forget_class 0 \
               --data_name cifar10 \
               --model_name AllCNN \
                --lr 0.01 \
                --epoch 25  \
                --batch_size 64  \
                --gpu_id 0 \
                --sgld \
                --do_unlearning \
                --name 'unlearn_cifar_class_0_multiple_gamma_point_0001_lamda_0' \
                --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
                --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
                --remain_reg 0.0001 \
                --lamda 0

```

## Requirements
- Conda (Anaconda/Miniconda)
- Python 3.8 or higher

### Data access
Model checkpoints original and retrain models on open source data can be downloaded from:

[https://drive.google.com/drive/folders/1PN5BYwHF3e0JIgM_EDvHRD5lUxhlHIGm?usp=drive_link
](https://drive.google.com/drive/folders/19w_w3P5MaowOTXCJOvRibW9tEMQaa4jh?usp=sharing)


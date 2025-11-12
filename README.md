# Targeted Unlearning with Perturbed Sign Gradient Methods

This repository implements a bilevel optimization approach to machine unlearning, enabling controlled forgetting of specific training samples while retaining performance on the remaining data. Our method outperforms standard baselines across five datasets (CIFAR-10, FashionMNIST, SVHN, Fundus, and Oculoplastics) and introduces:

- Tunable hyperparameters for remain/forget control (`λ`, `γ`, etc.)
- A novel model composition strategy that can outperform full retraining
- Unlearning under class-, device-, and diagnosis-based conditions (standard + selective)

![figure_1_homa](https://github.com/user-attachments/assets/56880ef7-40ef-48bf-8d87-fd9b5a4ab129)

---

## Quick Start

You can train a model or use an existing checkpoint. Example (training on FashionMNIST class 0):

```bash
python main.py --forget_class 0 \
               --data_name fashionmnist \
               --model_name AllCNN \
               --lr 0.01 \
               --epoch 15 \
               --batch_size 64 \
               --gpu_id 0 \
               --train
```

To perform unlearning using our method with SGLD-style regularization:

```bash
python main.py --forget_class 0 \
               --data_name cifar10 \
               --model_name AllCNN \
               --lr 0.01 \
               --epoch 25 \
               --batch_size 64 \
               --gpu_id 0 \
               --sgld \
               --do_unlearning \
               --specific_settings \
               --gamma 0.0001 \
               --lamda 0 \
               --remain_reg 0.0001 \
               --name 'unlearn_cifar10_class0' \
               --original_model 'model_checkpoints/...original_model.pth' \
               --retrain_model 'model_checkpoints/...retrain_model.pth'
```


For selective unlearning (e.g., forgetting samples with specific metadata):

```bash
python main.py --custom_unlearn \
               --to_forget 'Cirrus 800 FA' \
               --data_name fundus_3_class \
               --do_unlearning ...
```

Note that due to clinical data sharing requirements the above selective unlearning will not work. You will have to modify the cofe to build forget set based on clinical demographics.


Shell scripts we used for hp tuning and experimentation are found in SHELL_SCRIPTS

## Baselines 

Baselines were established using code found in baselines directory. 


## Helpful Parameters


--train, --retrain_only: for initial training or retraining without the forget set

--do_unlearning: enables boundary shrinkage unlearning

--sgld, --run_sota, --specific_settings: define the unlearning algorithm (useful in hp tuning and establshing baselines)

--custom_unlearn, --selective_unlearn: support metadata-driven or partial-sample forgetting

--specific_settings: define individual lambda and gamma values for unlearning 

--tsne_embeddings, --ensemble: analysis and ensemble mode support

## Data and Model Availability

[Original/retrain checkpoints for open-source datasets are available for download at this link](https://drive.google.com/drive/folders/1fBa1BhOXKdjBCWEM00OjZAeRsm3pUpjx)


## Cite Us

If you found this work useful, please consider citing:

@misc{nahass2025targetedunlearningusingperturbed,
      title={Targeted Unlearning Using Perturbed Sign Gradient Methods With Applications On Medical Images}, 
      author={George R. Nahass and Zhu Wang and Homa Rashidisabet and Won Hwa Kim and Sasha Hubschman and Jeffrey C. Peterson and Chad A. Purnell and Pete Setabutr and Ann Q. Tran and Darvin Yi and Sathya N. Ravi},
      year={2025},
      eprint={2505.21872},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2505.21872}, 
}

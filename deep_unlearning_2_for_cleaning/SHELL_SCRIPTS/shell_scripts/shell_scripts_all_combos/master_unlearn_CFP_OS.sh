#!/bin/sh

# CFP OS RESNET
# python main.py --forget_class 0 \
#                --data_name open_source \
#                --model_name resnet \
#                --dataset_dir ./fundus_open_source \
#                 --lr 0.001 \
#                 --epoch 15  \
#                 --batch_size 16  \
#                 --train \
#                 --gpu_id 1 \
#                 --name 'unlearn_CFP_OS_class_0_sota_sgld_closest_points_unlearning_NO_scale_DELETE' \
#                --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_20.pth' \
#                --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_20_class_0.pth' \



# python main.py --forget_class 0 \
#                --data_name open_source \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./fundus_open_source \
#                --batch_size 16 \
#                --gpu_id 1 \
#                 --sgld \
#                 --closest_points \
#                 --run_sota \
#                 --do_unlearning \
#                --name  'unlearn_CFP_OS_class_0_sota_sgld_closest_points_unlearning_NO_scale' \
#                --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_15.pth' \
#                --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_class_0_15.pth' 


# python main.py --forget_class 0 \
#                --data_name open_source \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./fundus_open_source \
#                --batch_size 16 \
#                --gpu_id 1 \
#                 --closest_points \
#                 --run_sota \
#                 --do_unlearning \
#                 --scaling inverse \
#                --name  'unlearn_CFP_OS_class_0_sota_sgld_closest_points_unlearning_INVERSE_scale' \
#                --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_15.pth' \
#                --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_class_0_15.pth' 


# python main.py --forget_class 0 \
#                --data_name open_source \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./fundus_open_source \
#                --batch_size 16 \
#                --gpu_id 1 \
#                 --closest_points \
#                 --run_sota \
#                 --do_unlearning \
#                 --scaling exponential_decay \
#                --name  'unlearn_CFP_OS_class_0_sota_sgld_closest_points_unlearning_EXPONENTIAL_scale' \
#                --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_15.pth' \
#                --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_class_0_15.pth' 


python main.py --forget_class 0 \
               --data_name open_source \
               --model_name resnet \
               --lr 0.001 \
               --epoch 15  \
               --dataset_dir ./fundus_open_source \
               --batch_size 16 \
               --gpu_id 1 \
                --closest_points \
                --run_sota \
                --do_unlearning \
                --scaling linear_decay \
               --name  'unlearn_CFP_OS_class_0_sota_sgld_closest_points_unlearning_LINEAR_scale' \
               --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_15.pth' \
               --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_class_0_15.pth' 


# # CFP OS ViT
# python main.py --method boundary_shrink \
#                --forget_class 1 \
#                --data_name open_source --model_name vit \
#                 --lr 0.001 --epoch 15  --dataset_dir ./fundus_open_source \
#                 --batch_size 16 --gpu_id 2 \
#                 --original_model 'model_checkpoints/vit_open_source_fundus/vitopen_source_original_model_15.pth' \
#                 --retrain_model  'model_checkpoints/vit_open_source_fundus/vitopen_source_retrain_model_class_1_13.pth' \
#                 --sgld 
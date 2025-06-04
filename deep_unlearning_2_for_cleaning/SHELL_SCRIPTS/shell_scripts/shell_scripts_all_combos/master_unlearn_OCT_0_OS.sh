#!/bin/sh


# OCT OS RESNET
# python main.py --forget_class 0 \
#                --data_name oct_4_class \
#                --model_name resnet \
#                --dataset_dir ./oct_open_source \
#                 --lr 0.001 \
#                 --epoch 15  \
#                 --batch_size 16  \
#                 --retrain_only \
#                 --gpu_id 2 \
#                 --name 'unlearn_CFP_OS_class_0_sota_sgld_closest_points_unlearning_NO_scale_DELETE' \
#                 --original_model 'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_original_model_15.pth' \
                # --retrain_model  'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_retrain_model_class_0_15.pth' 


python main.py --forget_class 0 \
               --data_name oct_4_class \
               --model_name resnet \
               --lr 0.001 \
               --epoch 15  \
               --dataset_dir ./oct_open_source \
               --batch_size 16 \
               --gpu_id 2 \
                --sgld \
                --closest_points \
                --run_sota \
                --do_unlearning \
               --name  'unlearn_OCT_OS_class_0_sota_sgld_closest_points_unlearning_NO_scale' \
                --original_model 'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_original_model_15.pth' \
                --retrain_model  'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_retrain_model_class_0_15.pth' 


python main.py --forget_class 0 \
               --data_name oct_4_class \
               --model_name resnet \
               --lr 0.001 \
               --epoch 15  \
               --dataset_dir ./oct_open_source \
               --batch_size 16 \
               --gpu_id 2 \
                --closest_points \
                --run_sota \
                --do_unlearning \
                --scaling inverse \
               --name  'unlearn_OCT_OS_class_0_sota_sgld_closest_points_unlearning_INVERSE_scale' \
                --original_model 'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_original_model_15.pth' \
                --retrain_model  'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_retrain_model_class_0_15.pth' 


python main.py --forget_class 0 \
               --data_name oct_4_class \
               --model_name resnet \
               --lr 0.001 \
               --epoch 15  \
               --dataset_dir ./oct_open_source \
               --batch_size 16 \
               --gpu_id 2 \
               --closest_points \
               --run_sota \
               --do_unlearning \
               --scaling exponential_decay \
               --name  'unlearn_OCT_OS_class_0_sota_sgld_closest_points_unlearning_EXPONENTIAL_scale' \
                --original_model 'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_original_model_15.pth' \
                --retrain_model  'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_retrain_model_class_0_15.pth' 

python main.py --forget_class 0 \
               --data_name oct_4_class \
               --model_name resnet \
               --lr 0.001 \
               --epoch 15  \
               --dataset_dir ./oct_open_source \
               --batch_size 16 \
               --gpu_id 2 \
               --closest_points \
               --run_sota \
               --do_unlearning \
               --scaling linear_decay \
               --name  'unlearn_OCT_OS_class_0_sota_sgld_closest_points_unlearning_LINEAR_scale' \
                --original_model 'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_original_model_15.pth' \
                --retrain_model  'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_retrain_model_class_0_15.pth' 



# python main.py --method boundary_shrink \
#                --forget_class 1 \
#                --data_name oct_4_class --model_name resnet \
#                 --lr 0.001 --epoch 15  --dataset_dir ./oct_open_source \
#                 --batch_size 16  \
#                 --original_model 'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_original_model_20.pth' \
#                 --retrain_model  'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_retrain_model_class_1_20.pth' \
#                 --sgld --gpu_id 0
                
# # OCT OS ViT
# python main.py --method boundary_shrink \
#                --forget_class 1 \
#                --data_name oct_4_class --model_name vit \
#                 --lr 0.001 --epoch 15  --dataset_dir ./oct_open_source \
#                 --batch_size 16  \
#                 --original_model 'model_checkpoints/vit_open_source_oct/vitoct_4_class_original_model_15.pth' \
#                 --retrain_model  'model_checkpoints/vit_open_source_oct/vitoct_4_class_retrain_model_class_1_15.pth' \
#                 --sgld
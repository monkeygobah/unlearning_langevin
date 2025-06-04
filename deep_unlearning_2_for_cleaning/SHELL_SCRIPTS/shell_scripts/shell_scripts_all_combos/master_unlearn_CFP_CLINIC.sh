#!/bin/sh

# CFP Clinical RESNET
python main.py --forget_class 0 \
               --data_name fundus_3_class \
               --model_name resnet \
               --dataset_dir ./fundus_big \
                --lr 0.001 \
                --epoch 40  \
                --batch_size 16  \
                --retrain_only \
                --gpu_id 0 \
                --name 'unlearn_CFP_CLINIC_class_0_sota_sgld_closest_points_unlearning_NO_scale_DELETE' \
                --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
                --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_40_16_bs_class_0.pth' 


# python main.py --forget_class 0 \
#                --data_name fundus_3_class \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 40  \
#                --dataset_dir ./fundus_big \
#                --batch_size 16 \
#                --gpu_id 0 \
#                 --sgld \
#                 --closest_points \
#                 --run_sota \
#                 --do_unlearning \
#                --name  'unlearn_CFP_CLINIC_class_0_sota_sgld_closest_points_unlearning_NO_scale_RERUN' \
#                 --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
#                 --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_class_0_40.pth' 


# python main.py --forget_class 0 \
#                --data_name fundus_3_class \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 40  \
#                --dataset_dir ./fundus_big \
#                --batch_size 16 \
#                --gpu_id 0 \
#                 --closest_points \
#                 --run_sota \
#                 --do_unlearning \
#                 --scaling inverse \
#                --name  'unlearn_CFP_CLINIC_class_0_sota_sgld_closest_points_unlearning_INVERSE_scale' \
#                 --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
#                 --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_class_0_40.pth' 


# python main.py --forget_class 0 \
#                --data_name fundus_3_class \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 40  \
#                --dataset_dir ./fundus_big \
#                --batch_size 16 \
#                --gpu_id 0 \
#                 --closest_points \
#                 --run_sota \
#                 --do_unlearning \
#                 --scaling exponential_decay \
#                --name  'unlearn_CFP_CLINIC_class_0_sota_sgld_closest_points_unlearning_EXPONENTIAL_scale' \
#                 --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
#                 --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_class_0_40.pth' 

# python main.py --forget_class 0 \
#                --data_name fundus_3_class \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 40  \
#                --dataset_dir ./fundus_big \
#                --batch_size 16 \
#                --gpu_id 0 \
#                 --closest_points \
#                 --run_sota \
#                 --do_unlearning \
#                 --scaling linear_decay \
#                --name  'unlearn_CFP_CLINIC_class_0_sota_sgld_closest_points_unlearning_LINEAR_scale' \
#                 --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
#                 --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_class_0_40.pth' 





# # python main.py --method boundary_shrink \
# #                --forget_class 0 \
# #                --data_name oct_4_class --model_name resnet \
# #                 --lr 0.001 --epoch 15  --dataset_dir ./oct_open_source \
# #                 --batch_size 16  \
# #                 --original_model 'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_original_model_20.pth' \
# #                 --retrain_model  'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_retrain_model_class_0_20.pth' \
# #                 --sgld --gpu_id 0
                



# ##### ViT EXPERIMENTS
# # CFP Clinical ViT
# # python main.py --method boundary_shrink \
# #                --forget_class 1 \
# #                --data_name fundus_3_class --model_name vit \
# #                 --lr 0.001 --epoch 15  --dataset_dir ./fundus_big \
# #                 --batch_size 16 --gpu_id 1 \
# #                 --original_model 'model_checkpoints/vit_clinincal_fundus/vitfundus_3_class_original_model_20.pth' \
# #                 --retrain_model  'model_checkpoints/vit_clinincal_fundus/vitfundus_3_class_retrain_model_class_1_15.pth' \
# #                 --sgld

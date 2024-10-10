

###################### FUNDUS



# python main.py --forget_class 0 \
#                --data_name open_source \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/fundus_open_source \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CFP_OS_class_0_multiple_gamma_remain_0_lamda_0001_DELETE' \
#                --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_20.pth' \
#                --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_20_class_0.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .0001


# python main.py --forget_class 0 \
#                --data_name open_source \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/fundus_open_source \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CFP_OS_class_0_multiple_gamma_remain_0_lamda_001' \
#                --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_20.pth' \
#                --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_20_class_0.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .001


# python main.py --forget_class 0 \
#                --data_name open_source \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/fundus_open_source \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CFP_OS_class_0_multiple_gamma_remain_0_lamda_01' \
#                --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_20.pth' \
#                --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_20_class_0.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .01


# python main.py --forget_class 0 \
#                --data_name open_source \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/fundus_open_source \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CFP_OS_class_0_multiple_gamma_remain_0_lamda_1' \
#                --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_20.pth' \
#                --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_20_class_0.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .1


# python main.py --forget_class 0 \
#                --data_name open_source \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/fundus_open_source \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CFP_OS_class_0_multiple_gamma_remain_0_lamda_0' \
#                --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_20.pth' \
#                --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_20_class_0.pth' \
#                 --remain_reg 0.0 \
#                 --lamda 0
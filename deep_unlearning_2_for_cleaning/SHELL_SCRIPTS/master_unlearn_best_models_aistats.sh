## CIFAR 

# Chen
# python main.py --forget_class 0 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --do_unlearning \
#                 --name 'CHEN_UNLEARN_FOR_BASELINE_CIFAR' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
#                 --remain_reg 0.0 \
#                 --run_sota


# # RAVI
# python main.py --forget_class 0 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --do_unlearning \
#                 --name 'RAVI_UNLEARN_FOR_BASELINE_CIFAR' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
#                 --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
#                 --remain_reg 0.0 \
#                 --specific_settings \
#                 --lamda .001 \
#                 --gamma .01


# ### CFP 
# # Chen
python main.py --forget_class 0 \
               --data_name open_source \
               --model_name resnet \
               --lr 0.001 \
               --epoch 15  \
               --dataset_dir ./data/fundus_open_source \
               --batch_size 16 \
               --gpu_id 0 \
               --do_unlearning \
               --name  'CHEN_UNLEARN_FOR_BASELINE_CFP' \
               --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_20.pth' \
               --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_20_class_0.pth' \
                --remain_reg 0.0 \
                --run_sota


# Ravi
python main.py --forget_class 0 \
               --data_name open_source \
               --model_name resnet \
               --lr 0.001 \
               --epoch 15  \
               --dataset_dir ./data/fundus_open_source \
               --batch_size 16 \
               --gpu_id 0 \
               --do_unlearning \
               --name  'RAVI_UNLEARN_FOR_BASELINE_CFP' \
               --original_model 'model_checkpoints/resnet_open_source_fundus/resnetopen_source_original_model_20.pth' \
               --retrain_model  'model_checkpoints/resnet_open_source_fundus/resnetopen_source_retrain_model_20_class_0.pth' \
                --remain_reg 0.0 \
                --specific_settings \
                --lamda .001 \
                --gamma .01



### MRI

# Chen
python main.py --forget_class 0 \
               --data_name mri \
               --model_name resnet \
               --dataset_dir data/mri_unlearn/ \
                --lr 0.001 \
                --epoch 15  \
                --batch_size 16  \
                --gpu_id 0 \
                --do_unlearning \
                --name 'CHEN_UNLEARN_FOR_BASELINE_MRI' \
                --original_model 'model_checkpoints/resnet_mri/resnetmri_original_model_15.pth' \
                --retrain_model 'model_checkpoints/resnet_mri/resnetmri_retrain_model_class_0_15.pth' \
                --remain_reg 0.0 \
                --run_sota


# RAVI
python main.py --forget_class 0 \
               --data_name mri \
               --model_name resnet \
               --dataset_dir data/mri_unlearn/ \
                --lr 0.001 \
                --epoch 15  \
                --batch_size 16  \
                --gpu_id 0 \
                --do_unlearning \
                --name 'RAVI_UNLEARN_FOR_BASELINE_MRI' \
                --original_model 'model_checkpoints/resnet_mri/resnetmri_original_model_15.pth' \
                --retrain_model 'model_checkpoints/resnet_mri/resnetmri_retrain_model_class_0_15.pth' \
                --remain_reg 0.0 \
                --specific_setting \
                --gamma 0.1 \
                --lamda .1



### FASHIONMNIST

# CHEN
python main.py --forget_class 0 \
                --data_name fashionmnist \
                --model_name AllCNN \
                --lr 0.01 \
                --epoch 15  \
                --batch_size 64  \
                --gpu_id 0 \
                --do_unlearning \
                --name "CHEN_UNLEARN_FOR_BASELINE_FASHION" \
                --original_model "model_checkpoints/allcnn_fashion/AllCNNfashionmnist_original_model_15.pth" \
                --retrain_model "model_checkpoints/allcnn_fashion/AllCNNfashionmnist_retrain_model_class_0_15.pth" \
                --remain_reg 0.0 \
                --run_sota


# RAVI
python main.py --forget_class 0 \
                --data_name fashionmnist \
                --model_name AllCNN \
                --lr 0.01 \
                --epoch 15  \
                --batch_size 64  \
                --gpu_id 0 \
                --do_unlearning \
                --name "RAVI_UNLEARN_FOR_BASELINE_FASHION" \
                --original_model "model_checkpoints/allcnn_fashion/AllCNNfashionmnist_original_model_15.pth" \
                --retrain_model "model_checkpoints/allcnn_fashion/AllCNNfashionmnist_retrain_model_class_0_15.pth" \
                --remain_reg 0.0 \
                --specific_setting \
                --gamma 0.0001 \
                --lamda 0


#### MED MNIST
# CHEN
python main.py --forget_class 0 \
                --data_name medmnist \
                --model_name AllCNN \
                --lr 0.01 \
                --epoch 15  \
                --batch_size 64  \
                --gpu_id 0 \
                --do_unlearning \
                --name "CHEN_UNLEARN_FOR_BASELINE_MEDMNIST" \
                --original_model "model_checkpoints/allcnn_medmnist/AllCNNmedmnist_original_model_15.pth" \
                --retrain_model "model_checkpoints/allcnn_medmnist/AllCNNmedmnist_retrain_model_class_0_15.pth" \
                --remain_reg 0.0 \
                --run_sota

# RAVI
python main.py --forget_class 0 \
                --data_name medmnist \
                --model_name AllCNN \
                --lr 0.01 \
                --epoch 15  \
                --batch_size 64  \
                --gpu_id 0 \
                --do_unlearning \
                --name "RAVI_UNLEARN_FOR_BASELINE_MEDMNIST" \
                --original_model "model_checkpoints/allcnn_medmnist/AllCNNmedmnist_original_model_15.pth" \
                --retrain_model "model_checkpoints/allcnn_medmnist/AllCNNmedmnist_retrain_model_class_0_15.pth" \
                --remain_reg 0.0 \
                --specific_setting \
                --gamma 0.0001 \
                --lamda 0
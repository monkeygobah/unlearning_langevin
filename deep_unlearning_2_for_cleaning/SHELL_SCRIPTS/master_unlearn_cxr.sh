# # CXR UNLEARN
# python main.py --forget_class 2  \
#                --data_name cxr \
#                --model_name resnet \
#                --dataset_dir data/xray_unlearn_oversample/ \
#                 --lr 0.001 \
#                 --epoch 15  \
#                 --batch_size 16  \
#                 --retrain_only \
#                 --gpu_id 0 \
#                 --name 'unlearn_CXR_OS_class_2_sota_sgld_closest_points_unlearning_DELETE' \
#                 --original_model 'model_checkpoints/resnet_chest_xray/resnetcxr_original_model_15.pth'


# python main.py --forget_class 0  \
#                --data_name cxr \
#                --model_name resnet \
#                --dataset_dir data/xray_unlearn_oversample/ \
#                 --lr 0.001 \
#                 --epoch 10  \
#                 --batch_size 16  \
#                 --retrain_only \
#                 --gpu_id 0 \
#                 --name 'unlearn_CXR_OS_class_2_sota_sgld_closest_points_unlearning_DELETE' \
#                 --original_model 'model_checkpoints/resnet_chest_xray/resnetcxr_original_model_15.pth'

                
# python main.py --forget_class 1  \
#                --data_name cxr \
#                --model_name resnet \
#                --dataset_dir data/xray_unlearn_oversample/ \
#                 --lr 0.001 \
#                 --epoch 10  \
#                 --batch_size 16  \
#                 --retrain_only \
#                 --gpu_id 0 \
#                 --name 'unlearn_CXR_OS_class_2_sota_sgld_closest_points_unlearning_DELETE' \
#                 --original_model 'model_checkpoints/resnet_chest_xray/resnetcxr_original_model_15.pth'


# python main.py --forget_class 3  \
#                --data_name cxr \
#                --model_name resnet \
#                --dataset_dir data/xray_unlearn_oversample/ \
#                 --lr 0.001 \
#                 --epoch 10  \
#                 --batch_size 16  \
#                 --retrain_only \
#                 --gpu_id 0 \
#                 --name 'unlearn_CXR_OS_class_2_sota_sgld_closest_points_unlearning_DELETE' \
#                 --original_model 'model_checkpoints/resnet_chest_xray/resnetcxr_original_model_15.pth'



# python main.py --forget_class 0 \
#                --data_name cxr \
#                --model_name resnet \
#                --dataset_dir data/xray_unlearn_oversample/ \
#                 --lr 0.001 \
#                 --epoch 10  \
#                 --batch_size 16  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --closest_points \
#                 --run_sota \
#                 --do_unlearning \
#                 --name 'unlearn_CXR_class_0_sota_sgld_closest_points_unlearning_NO_scale' \
#                 --original_model 'model_checkpoints/resnet_chest_xray/resnetcxr_original_model_10.pth' \
#                 --retrain_model 'model_checkpoints/resnet_chest_xray/resnetcxr_retrain_model_class_0_10.pth' 

python main.py --forget_class 1 \
               --data_name cxr \
               --model_name resnet \
               --dataset_dir data/xray_unlearn_oversample/ \
                --lr 0.001 \
                --epoch 10  \
                --batch_size 16  \
                --gpu_id 0 \
                --sgld \
                --closest_points \
                --run_sota \
                --do_unlearning \
                --name 'unlearn_CXR_class_1_sota_sgld_closest_points_unlearning_NO_scale' \
                --original_model 'model_checkpoints/resnet_chest_xray/resnetcxr_original_model_10.pth' \
                --retrain_model 'model_checkpoints/resnet_chest_xray/resnetcxr_retrain_model_class_1_10.pth' 



# python main.py --forget_class 2 \
#                --data_name cxr \
#                --model_name resnet \
#                --dataset_dir data/xray_unlearn_oversample/ \
#                 --lr 0.001 \
#                 --epoch 30  \
#                 --batch_size 16  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --closest_points \
#                 --run_sota \
#                 --do_unlearning \
#                 --name 'unlearn_CXR_class_2_sota_sgld_closest_points_unlearning_NO_scale' \
#                 --original_model 'model_checkpoints/resnet_chest_xray/resnetcxr_original_model_10.pth' \
#                 --retrain_model 'model_checkpoints/resnet_chest_xray/resnetcxr_retrain_model_class_2_10.pth' 

# python main.py --forget_class 3 \
#                --data_name cxr \
#                --model_name resnet \
#                --dataset_dir data/xray_unlearn_oversample/ \
#                 --lr 0.001 \
#                 --epoch 10  \
#                 --batch_size 16  \
#                 --gpu_id 0 \
#                 --sgld \
#                 --closest_points \
#                 --run_sota \
#                 --do_unlearning \
#                 --name 'unlearn_CXR_class_3_sota_sgld_closest_points_unlearning_NO_scale' \
#                 --original_model 'model_checkpoints/resnet_chest_xray/resnetcxr_original_model_10.pth' \
#                 --retrain_model 'model_checkpoints/resnet_chest_xray/resnetcxr_retrain_model_class_3_10.pth' 

python main.py --forget_class 8 \
               --data_name cifar10 \
               --model_name AllCNN \
                --lr 0.01 \
                --epoch 25  \
                --batch_size 64  \
                --gpu_id 0 \
                --sgld \
                --closest_points \
                --run_sota \
                --do_unlearning \
                --name 'unlearn_cifar_class_8_sota_sgld_closest_points_unlearning_NO_scale' \
                --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
                --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_8_25.pth' 
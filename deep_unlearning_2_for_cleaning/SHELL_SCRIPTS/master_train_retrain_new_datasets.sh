python main.py --forget_class 0 \
                --data_name fashionmnist \
                --model_name AllCNN \
                --lr 0.01 \
                --epoch 15  \
                --batch_size 64  \
                --gpu_id 0 \
                --train \
                --name "unlearn_fashionmnist_class_0_sota_sgld_closest_points_unlearning_NO_scale" 

# python main.py --forget_class 0 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 25  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --retrain_only \
#                 --name 'unlearn_cifar_class_0_sota_sgld_closest_points_unlearning_NO_scale' \
#                 --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' 

# python main.py --forget_class 4 \
#                 --data_name medmnist \
#                 --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 15  \
#                 --batch_size 64  \
#                 --gpu_id 0 \
#                 --retrain_only \
#                 --name "unlearn_medmnist_class_4_sota_sgld_closest_points_unlearning_NO_scale"  \
#                 --original_model "model_checkpoints/allcnn_medmnist/AllCNNmedmnist_original_model_15.pth"



# # FashionMNIST
# for i in {1..9}; do
#     python main.py --forget_class $i \
#                    --data_name fashionmnist \
#                    --model_name AllCNN \
#                    --lr 0.01 \
#                    --epoch 15  \
#                    --batch_size 64  \
#                    --gpu_id 0 \
#                    --retrain_only \
#                    --name "unlearn_fashionmnist_class_${i}_sota_sgld_closest_points_unlearning_NO_scale" \
#                    --original_model "model_checkpoints/allcnn_fashion/AllCNNfashionmnist_original_model_15.pth"
# done

# # medmnist
# for i in {1..9}; do
#     python main.py --forget_class $i \
#                    --data_name medmnist \
#                    --model_name AllCNN \
#                    --lr 0.01 \
#                    --epoch 15  \
#                    --batch_size 64  \
#                    --gpu_id 0 \
#                    --retrain_only \
#                    --name "unlearn_medmnist_class_${i}_sota_sgld_closest_points_unlearning_NO_scale" \
#                    --original_model "model_checkpoints/allcnn_medmnist/AllCNNmedmnist_original_model_15.pth"
# done

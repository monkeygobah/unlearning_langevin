python main.py --method boundary_shrink \
               --forget_class 2 \
               --data_name mnist --model_name MNISTNet \
                --lr 0.001 --epoch 50  \
                --batch_size 16  \
                --original_model 'model_checkpoints/MNISTNetmnist_original_model_50.pth' \
                --retrain_model 'model_checkpoints/MNISTNetmnist_retrain_model_class_2_50.pth' \
                --sgld --gpu_id 1 --run_sota --closest_points \
                --name 'unlearn_mnist_sota_sgld_closest_points_all_dists_gammas_true_unlearning'
                 
                



# python main.py --method boundary_shrink \
#                --forget_class 2 \
#                --data_name mnist --model_name MNISTNet \
#                 --lr 0.001 --epoch 20  \
#                 --batch_size 16  \
#                 --train \
#                 --original_model 'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_original_model_20.pth' \
#                 --retrain_model  'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_retrain_model_class_1_20.pth' \
#                 --sgld --gpu_id 1 \ 
#                 --run_sota
                
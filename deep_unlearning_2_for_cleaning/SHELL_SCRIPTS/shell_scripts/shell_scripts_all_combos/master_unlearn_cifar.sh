# python main.py --forget_class 2 \
#                --data_name cifar10 \
#                --model_name AllCNN \
#                 --lr 0.01 \
#                 --epoch 60  \
#                 --batch_size 64  \
#                 --train \
#                 --gpu_id 1 \
#                 --name 'unlearn_cifar_class_2_sota_sgld_closest_points_unlearning_TRAINING_CAN_DELETE' \
#                 --retrain_model 'model_checkpoints/AllCNNcifar10_retrain_model_class_2_60.pth' \
#                 --original_model 'model_checkpoints/AllCNNcifar10_original_model_60.pth' \


python main.py --forget_class 2 \
               --data_name cifar10 \
               --model_name AllCNN \
                --lr 0.01 \
                --epoch 60  \
                --batch_size 64  \
                --gpu_id 1 \
                --sgld \
                --closest_points \
                --run_sota \
                --do_unlearning \
                --name 'unlearn_cifar_class_2_sota_sgld_closest_points_unlearning_NO_scale' \
                --retrain_model 'model_checkpoints/AllCNNcifar10_retrain_model_class_2_60.pth' \
                --original_model 'model_checkpoints/AllCNNcifar10_original_model_60.pth' \


python main.py --forget_class 2 \
               --data_name cifar10 \
               --model_name AllCNN \
                --lr 0.01 \
                --epoch 60  \
                --batch_size 64  \
                --gpu_id 1 \
                --closest_points \
                --run_sota \
                --do_unlearning \
                --scaling inverse \
                --name 'unlearn_cifar_class_2_sota_sgld_closest_points_unlearning_INVERSE_scale' \
                --retrain_model 'model_checkpoints/AllCNNcifar10_retrain_model_class_2_60.pth' \
                --original_model 'model_checkpoints/AllCNNcifar10_original_model_60.pth' \

python main.py --forget_class 2 \
               --data_name cifar10 \
               --model_name AllCNN \
                --lr 0.01 \
                --epoch 60  \
                --batch_size 64  \
                --gpu_id 1 \
                --closest_points \
                --run_sota \
                --do_unlearning \
                --scaling exponential_decay \
                --name 'unlearn_cifar_class_2_sota_sgld_closest_points_unlearning_EXPONENTIAL_scale' \
                --retrain_model 'model_checkpoints/AllCNNcifar10_retrain_model_class_2_60.pth' \
                --original_model 'model_checkpoints/AllCNNcifar10_original_model_60.pth' \

python main.py --forget_class 2 \
               --data_name cifar10 \
               --model_name AllCNN \
                --lr 0.01 \
                --epoch 60  \
                --batch_size 64  \
                --gpu_id 1 \
                --closest_points \
                --run_sota \
                --do_unlearning \
                --scaling linear_decay \
                --name 'unlearn_cifar_class_2_sota_sgld_closest_points_unlearning_LINEAR_scale' \
                --retrain_model 'model_checkpoints/AllCNNcifar10_retrain_model_class_2_60.pth' \
                --original_model 'model_checkpoints/AllCNNcifar10_original_model_60.pth' \

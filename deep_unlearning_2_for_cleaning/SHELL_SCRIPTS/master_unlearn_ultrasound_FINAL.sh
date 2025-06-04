################ ULTRASOUND

### Class 0 

python main.py --forget_class 0 \
               --data_name ultrasound \
               --model_name resnet \
                --lr 0.001 \
               --epoch 30  \
               --dataset_dir data/ultrasound_unlearn_oversample/ \
               --batch_size 16 \
               --gpu_id 0 \
               --sgld \
               --do_unlearning \
               --name  'unlearn_ULTRASOUND_CLASS_0_LAMDA_0001' \
                --original_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_original_model_30.pth' \
                --retrain_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_retrain_model_class_0_30.pth' \
                --remain_reg 0.0 \
                --lamda .0001


# python main.py --forget_class 0 \
#                --data_name ultrasound \
#                --model_name resnet \
#                 --lr 0.001 \
#                --epoch 30  \
#                --dataset_dir data/ultrasound_unlearn_oversample/ \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_ULTRASOUND_CLASS_0_LAMDA_001' \
#                 --original_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_original_model_30.pth' \
#                 --retrain_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_retrain_model_class_0_30.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .001


# python main.py --forget_class 0 \
#                --data_name ultrasound \
#                --model_name resnet \
#                 --lr 0.001 \
#                --epoch 30  \
#                --dataset_dir data/ultrasound_unlearn_oversample/ \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_ULTRASOUND_CLASS_0_LAMDA_01' \
#                 --original_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_original_model_30.pth' \
#                 --retrain_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_retrain_model_class_0_30.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .01


# python main.py --forget_class 0 \
#                --data_name ultrasound \
#                --model_name resnet \
#                 --lr 0.001 \
#                --epoch 30  \
#                --dataset_dir data/ultrasound_unlearn_oversample/ \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_ULTRASOUND_CLASS_0_LAMDA_1' \
#                 --original_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_original_model_30.pth' \
#                 --retrain_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_retrain_model_class_0_30.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .1


# python main.py --forget_class 0 \
#                --data_name ultrasound \
#                --model_name resnet \
#                 --lr 0.001 \
#                --epoch 30  \
#                --dataset_dir data/ultrasound_unlearn_oversample/ \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_ULTRASOUND_CLASS_0_LAMDA_0' \
#                 --original_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_original_model_30.pth' \
#                 --retrain_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_retrain_model_class_0_30.pth' \
#                 --remain_reg 0.0 \
#                 --lamda 0






### Class 2 

python main.py --forget_class 2 \
               --data_name ultrasound \
               --model_name resnet \
                --lr 0.001 \
               --epoch 30  \
               --dataset_dir data/ultrasound_unlearn_oversample/ \
               --batch_size 16 \
               --gpu_id 0 \
               --sgld \
               --do_unlearning \
               --name  'unlearn_ULTRASOUND_CLASS_2_LAMDA_0001' \
                --original_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_original_model_30.pth' \
                --retrain_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_retrain_model_class_2_30.pth' \
                --remain_reg 0.0 \
                --lamda .0001


# python main.py --forget_class 2 \
#                --data_name ultrasound \
#                --model_name resnet \
#                 --lr 0.001 \
#                --epoch 30  \
#                --dataset_dir data/ultrasound_unlearn_oversample/ \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_ULTRASOUND_CLASS_2_LAMDA_001' \
#                 --original_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_original_model_30.pth' \
#                 --retrain_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_retrain_model_class_2_30.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .001


# python main.py --forget_class 2 \
#                --data_name ultrasound \
#                --model_name resnet \
#                 --lr 0.001 \
#                --epoch 30  \
#                --dataset_dir data/ultrasound_unlearn_oversample/ \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_ULTRASOUND_CLASS_2_LAMDA_01' \
#                 --original_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_original_model_30.pth' \
#                 --retrain_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_retrain_model_class_2_30.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .01


# python main.py --forget_class 2 \
#                --data_name ultrasound \
#                --model_name resnet \
#                 --lr 0.001 \
#                --epoch 30  \
#                --dataset_dir data/ultrasound_unlearn_oversample/ \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_ULTRASOUND_CLASS_2_LAMDA_1' \
#                 --original_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_original_model_30.pth' \
#                 --retrain_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_retrain_model_class_2_30.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .1


# python main.py --forget_class 2 \
#                --data_name ultrasound \
#                --model_name resnet \
#                 --lr 0.001 \
#                --epoch 30  \
#                --dataset_dir data/ultrasound_unlearn_oversample/ \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_ULTRASOUND_CLASS_2_LAMDA_0' \
#                 --original_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_original_model_30.pth' \
#                 --retrain_model 'model_checkpoints/resnet_ultrasound/resnetultrasound_retrain_model_class_2_30.pth' \
#                 --remain_reg 0.0 \
#                 --lamda 0



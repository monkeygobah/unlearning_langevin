### OCT RETRAIN

# python main.py --forget_class 0 \
#                --data_name oct_4_class \
#                --model_name resnet \
#               --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/oct_open_source \
#                --batch_size 8 \
#                --gpu_id 0 \
#                --retrain_only \
#                --name  'unlearn_CLINCAL_OCT_CLASS_0_LAMDA_0_RETRAIN' \
#                 --original_model 'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_original_model_20.pth' \
#                 --retrain_model  'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_retrain_model_class_0_20.pth' \
#                 --remain_reg 0.0 \
#                 --lamda 0


# python main.py --forget_class 1 \
#                --data_name oct_4_class \
#                --model_name resnet \
#               --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/oct_open_source \
#                --batch_size 8 \
#                --gpu_id 0 \
#                --retrain_only \
#                --name  'unlearn_CLINCAL_OCT_CLASS_1_LAMDA_0_RETRAIN' \
#                 --original_model 'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_original_model_20.pth' \
#                 --retrain_model  'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_retrain_model_class_1_20.pth' \
#                 --remain_reg 0.0 \
#                 --lamda 0



###################### OCT CLASS 0 UNLEARN

python main.py --forget_class 0 \
               --data_name oct_4_class \
               --model_name resnet \
              --lr 0.001 \
               --epoch 15  \
               --dataset_dir ./data/oct_open_source \
               --batch_size 8 \
               --gpu_id 0 \
               --sgld \
               --do_unlearning \
               --name  'unlearn_CLINCAL_OCT_CLASS_0_LAMDA_0' \
                --original_model 'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_original_model_20.pth' \
                --retrain_model  'model_checkpoints/resnet_open_source_oct/resnet_oct_4_class_retrain_unlearn_CLINCAL_OCT_CLASS_0_LAMDA_0_RETRAIN_14.pth' \
                --remain_reg 0.0 \
                --lamda 0


# python main.py --forget_class 0 \
#                --data_name oct_4_class \
#                --model_name resnet \
#               --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/oct_open_source \
#                --batch_size 8 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CLINCAL_OCT_CLASS_0_LAMDA_0001' \
#                 --original_model 'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_original_model_20.pth' \
#                 --retrain_model  'model_checkpoints/resnet_open_source_oct/resnet_oct_4_class_retrain_unlearn_CLINCAL_OCT_CLASS_0_LAMDA_0_RETRAIN_14.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .0001


# python main.py --forget_class 0 \
#                --data_name oct_4_class \
#                --model_name resnet \
#               --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/oct_open_source \
#                --batch_size 8 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CLINCAL_OCT_CLASS_0_LAMDA_001' \
#                 --original_model 'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_original_model_20.pth' \
#                 --retrain_model  'model_checkpoints/resnet_open_source_oct/resnet_oct_4_class_retrain_unlearn_CLINCAL_OCT_CLASS_0_LAMDA_0_RETRAIN_14.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .001


# python main.py --forget_class 0 \
#                --data_name oct_4_class \
#                --model_name resnet \
#               --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/oct_open_source \
#                --batch_size 8 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CLINCAL_OCT_CLASS_0_LAMDA_01' \
#                 --original_model 'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_original_model_20.pth' \
#                 --retrain_model  'model_checkpoints/resnet_open_source_oct/resnet_oct_4_class_retrain_unlearn_CLINCAL_OCT_CLASS_0_LAMDA_0_RETRAIN_14.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .01


# python main.py --forget_class 0 \
#                --data_name oct_4_class \
#                --model_name resnet \
#               --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/oct_open_source \
#                --batch_size 8 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CLINCAL_OCT_CLASS_0_LAMDA_1' \
#                 --original_model 'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_original_model_20.pth' \
#                 --retrain_model  'model_checkpoints/resnet_open_source_oct/resnet_oct_4_class_retrain_unlearn_CLINCAL_OCT_CLASS_0_LAMDA_0_RETRAIN_14.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .1



###################### OCT CLASS 1 UNLEARN

python main.py --forget_class 1 \
               --data_name oct_4_class \
               --model_name resnet \
              --lr 0.001 \
               --epoch 15  \
               --dataset_dir ./data/oct_open_source \
               --batch_size 8 \
               --gpu_id 0 \
               --sgld \
               --do_unlearning \
               --name  'unlearn_CLINCAL_OCT_CLASS_1_LAMDA_0' \
                --original_model 'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_original_model_20.pth' \
                --retrain_model  'model_checkpoints/resnet_open_source_oct/resnet_oct_4_class_retrain_unlearn_CLINCAL_OCT_CLASS_1_LAMDA_0_RETRAIN_15.pth' \
                --remain_reg 0.0 \
                --lamda 0

        

# python main.py --forget_class 1 \
#                --data_name oct_4_class \
#                --model_name resnet \
#               --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/oct_open_source \
#                --batch_size 8 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CLINCAL_OCT_CLASS_1_LAMDA_0001' \
#                 --original_model 'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_original_model_20.pth' \
#                 --retrain_model  'model_checkpoints/resnet_open_source_oct/resnet_oct_4_class_retrain_unlearn_CLINCAL_OCT_CLASS_1_LAMDA_0_RETRAIN_15.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .0001


# python main.py --forget_class 1 \
#                --data_name oct_4_class \
#                --model_name resnet \
#               --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/oct_open_source \
#                --batch_size 8 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CLINCAL_OCT_CLASS_1_LAMDA_001' \
#                 --original_model 'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_original_model_20.pth' \
#                 --retrain_model  'model_checkpoints/resnet_open_source_oct/resnet_oct_4_class_retrain_unlearn_CLINCAL_OCT_CLASS_1_LAMDA_0_RETRAIN_15.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .001


# python main.py --forget_class 1 \
#                --data_name oct_4_class \
#                --model_name resnet \
#               --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/oct_open_source \
#                --batch_size 8 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CLINCAL_OCT_CLASS_1_LAMDA_01' \
#                 --original_model 'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_original_model_20.pth' \
#                 --retrain_model  'model_checkpoints/resnet_open_source_oct/resnet_oct_4_class_retrain_unlearn_CLINCAL_OCT_CLASS_1_LAMDA_0_RETRAIN_15.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .01


# python main.py --forget_class 1 \
#                --data_name oct_4_class \
#                --model_name resnet \
#               --lr 0.001 \
#                --epoch 15  \
#                --dataset_dir ./data/oct_open_source \
#                --batch_size 8 \
#                --gpu_id 0 \
#                --sgld \
#                --do_unlearning \
#                --name  'unlearn_CLINCAL_OCT_CLASS_1_LAMDA_1' \
#                 --original_model 'model_checkpoints/resnet_open_source_oct/resnetoct_4_class_original_model_20.pth' \
#                 --retrain_model  'model_checkpoints/resnet_open_source_oct/resnet_oct_4_class_retrain_unlearn_CLINCAL_OCT_CLASS_1_LAMDA_0_RETRAIN_15.pth' \
#                 --remain_reg 0.0 \
#                 --lamda .1


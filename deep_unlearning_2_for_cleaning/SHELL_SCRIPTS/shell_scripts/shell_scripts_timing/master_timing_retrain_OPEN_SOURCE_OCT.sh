




# python main.py --forget_class 0 \
#                --data_name oct_4_class \
#                --model_name resnet \
#                --dataset_dir ./oct_open_source \
#                --gpu_id 0 \
#                --do_unlearning \
#                --specific_settings \
#                --closest_points \
#                --lamda .1 \
#                --dist normal \
#                --name  'unlearn_CFP_CLINIC_specific_settings' \
#                --retrain_only \
#                 --lr 0.001 \
#                 --epoch 15  \
#                 --batch_size 16  \
#                --percent_to_forget 1 \
#                --original_model 'model_checkpoints/resnet_open_source_oct_artemis/resnetoct_4_class_original_model_15.pth' \
#                --retrain_model  'model_checkpoints/resnet_open_source_oct_artemis/resnetoct_4_class_retrain_model_class_0_15.pth' 

python main.py --forget_class 0 \
               --data_name oct_4_class \
               --model_name resnet \
               --dataset_dir ./oct_open_source \
               --gpu_id 0 \
               --do_unlearning \
               --specific_settings \
               --closest_points \
               --lamda .1 \
               --dist normal \
               --name  'unlearn_CFP_CLINIC_specific_settings' \
                --lr 0.001 \
                --epoch 15  \
                --batch_size 16  \
               --percent_to_forget 1 \
               --original_model 'model_checkpoints/resnet_open_source_oct_artemis/resnetoct_4_class_original_model_15.pth' \
               --retrain_model  'model_checkpoints/resnet_open_source_oct_artemis/resnetoct_4_class_retrain_model_class_0_15.pth' 


python main.py --forget_class 0 \
               --data_name oct_4_class \
               --model_name resnet \
               --dataset_dir ./oct_open_source \
               --gpu_id 0 \
               --do_unlearning \
               --specific_settings \
               --closest_points \
               --lamda .1 \
               --dist normal \
               --name  'unlearn_CFP_CLINIC_specific_settings' \
                --lr 0.001 \
                --epoch 15  \
                --batch_size 16  \
               --percent_to_forget .5 \
               --original_model 'model_checkpoints/resnet_open_source_oct_artemis/resnetoct_4_class_original_model_15.pth' \
               --retrain_model  'model_checkpoints/resnet_open_source_oct_artemis/resnetoct_4_class_retrain_model_class_0_15.pth' 



python main.py --forget_class 0 \
               --data_name oct_4_class \
               --model_name resnet \
               --dataset_dir ./oct_open_source \
               --gpu_id 0 \
               --do_unlearning \
               --specific_settings \
               --closest_points \
               --lamda .1 \
               --dist normal \
               --name  'unlearn_CFP_CLINIC_specific_settings' \
                --lr 0.001 \
                --epoch 15  \
                --batch_size 16  \
               --percent_to_forget .25 \
               --original_model 'model_checkpoints/resnet_open_source_oct_artemis/resnetoct_4_class_original_model_15.pth' \
               --retrain_model  'model_checkpoints/resnet_open_source_oct_artemis/resnetoct_4_class_retrain_model_class_0_15.pth' 



python main.py --forget_class 0 \
               --data_name oct_4_class \
               --model_name resnet \
               --dataset_dir ./oct_open_source \
               --gpu_id 0 \
               --do_unlearning \
               --specific_settings \
               --closest_points \
               --lamda .1 \
               --dist normal \
               --name  'unlearn_CFP_CLINIC_specific_settings' \
                --lr 0.001 \
                --epoch 15  \
                --batch_size 16  \
               --percent_to_forget .13 \
               --original_model 'model_checkpoints/resnet_open_source_oct_artemis/resnetoct_4_class_original_model_15.pth' \
               --retrain_model  'model_checkpoints/resnet_open_source_oct_artemis/resnetoct_4_class_retrain_model_class_0_15.pth' 


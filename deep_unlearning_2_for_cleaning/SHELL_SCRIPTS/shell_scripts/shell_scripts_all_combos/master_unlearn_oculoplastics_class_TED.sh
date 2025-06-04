# python main.py  --forget_class 2 \
#                 --data_name oculoplastic \
#                 --model_name resnet \
#                 --dataset_dir ./oculoplastic \
#                 --lr 0.001 \
#                 --epoch 30  \
#                 --batch_size 16  \
#                 --train \
#                 --gpu_id 0 \
#                 --name 'unlearn_class_TED_sota_sgld_closest_points_unlearning_NO_scale_DELETE' \
#                 --original_model 'model_checkpoints/resnet_oculoplastic/resnetoculoplastic_original_model_30.pth' \
#                 --retrain_model  'model_checkpoints/resnet_oculoplastic/resnetoculoplastic_retrain_model_mrd_1_30.pth' 


python main.py --forget_class 2 \
               --data_name oculoplastic \
               --model_name resnet \
               --lr 0.01 \
               --epoch 30  \
               --dataset_dir ./oculoplastic \
               --batch_size 16 \
               --gpu_id 0 \
               --sgld \
               --closest_points \
               --run_sota \
               --do_unlearning \
               --name 'unlearn_class_TED_sota_sgld_closest_points_unlearning_NO_scale' \
                --original_model 'model_checkpoints/resnet_oculoplastic/resnetoculoplastic_original_model_30_3_class.pth' \
                --retrain_model  'model_checkpoints/resnet_oculoplastic/resnetoculoplastic_retrain_model_class_2_30_3_class.pth' 

python main.py --forget_class 2 \
               --data_name oculoplastic \
               --model_name resnet \
               --lr 0.01 \
               --epoch 30  \
               --dataset_dir ./oculoplastic \
               --batch_size 16 \
               --gpu_id 0 \
                --closest_points \
                --run_sota \
                --do_unlearning \
                --scaling inverse \
                --name 'unlearn_class_TED_sota_sgld_closest_points_unlearning_INVERSE_scale' \
                --original_model 'model_checkpoints/resnet_oculoplastic/resnetoculoplastic_original_model_30_3_class.pth' \
                --retrain_model  'model_checkpoints/resnet_oculoplastic/resnetoculoplastic_retrain_model_class_2_30_3_class.pth' 


python main.py --forget_class 2 \
               --data_name oculoplastic \
               --model_name resnet \
               --lr 0.01 \
               --epoch 30  \
               --dataset_dir ./oculoplastic \
               --batch_size 16 \
               --gpu_id 0 \
                --closest_points \
                --run_sota \
                --do_unlearning \
                --scaling exponential_decay \
                --name 'unlearn_class_TED_sota_sgld_closest_points_unlearning_EXPONENTIAL_scale' \
                --original_model 'model_checkpoints/resnet_oculoplastic/resnetoculoplastic_original_model_30_3_class.pth' \
                --retrain_model  'model_checkpoints/resnet_oculoplastic/resnetoculoplastic_retrain_model_class_2_30_3_class.pth' 

python main.py --forget_class 2 \
               --data_name oculoplastic \
               --model_name resnet \
               --lr 0.01 \
               --epoch 30  \
               --dataset_dir ./oculoplastic \
               --batch_size 16 \
               --gpu_id 0 \
               --closest_points \
               --run_sota \
               --do_unlearning \
               --scaling linear_decay \
               --name 'unlearn_class_TED_sota_sgld_closest_points_unlearning_LINEAR_scale' \
                --original_model 'model_checkpoints/resnet_oculoplastic/resnetoculoplastic_original_model_30_3_class.pth' \
                --retrain_model  'model_checkpoints/resnet_oculoplastic/resnetoculoplastic_retrain_model_class_2_30_3_class.pth' 


python main.py  --forget_class 2 \
                --data_name oculoplastic \
                --model_name resnet \
                --dataset_dir ./oculoplastic \
                --lr 0.001 \
                --epoch 30  \
                --batch_size 16  \
                --gpu_id 0 \
                --name 'unlearn_class_TED_sota_sgld_closest_points_unlearning_NO_scale_DELETE' \
                --do_unlearning \
               --specific_settings \
               --sgld \
               --gamma .5 \
               --dist normal \
               --percent_to_forget .5 \
                --original_model 'model_checkpoints/resnet_oculoplastic/resnetoculoplastic_original_model_30_3_class.pth' \
                --retrain_model  'model_checkpoints/resnet_oculoplastic/resnetoculoplastic_retrain_model_class_2_30_3_class.pth' 
            #    --run_sota \

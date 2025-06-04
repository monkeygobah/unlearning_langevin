



python main.py --custom_unlearn \
               --to_forget 'Cirrus 800 FA' \
               --data_name fundus_3_class \
               --model_name resnet \
               --dataset_dir ./fundus_big \
               --gpu_id 0 \
               --do_unlearning \
               --name 'unlearn_CAMERA_CIRRUS_specific_settings_exp' \
               --specific_settings \
               --closest_points \
               --lamda .1 \
               --dist un \
               --scaling exponential_decay \
                --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
                --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_40_16_bs_class_cirrus.pth' \


# python main.py --custom_unlearn \
#                --to_forget 'Cirrus 800 FA' \
#                --data_name fundus_3_class \
#                --model_name resnet \
#                --dataset_dir ./fundus_big \
#                --gpu_id 0 \
#                --do_unlearning \
#                --name 'unlearn_CAMERA_CIRRUS_specific_settings_linear' \
#                --specific_settings \
#                --closest_points \
#                --lamda .1 \
#                --dist normal \
#                --scaling linear_decay \
#                 --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
#                 --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_40_16_bs_class_cirrus.pth' \

# python main.py --custom_unlearn \
#                --to_forget 'Cirrus 800 FA' \
#                --data_name fundus_3_class \
#                --model_name resnet \
#                --dataset_dir ./fundus_big \
#                --gpu_id 0 \
#                --do_unlearning \
#                --name 'unlearn_CAMERA_CIRRUS_specific_settings_inverse' \
#                --specific_settings \
#                --sgld \
#                --gamma .1 \
#                --dist normal \
#                --scaling inverse \
#                 --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
#                 --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_40_16_bs_class_cirrus.pth' \











# python main.py --forget_class 0 \
#                --data_name fundus_3_class \
#                --model_name resnet \
#                --dataset_dir ./fundus_big \
#                 --sgld \
#                --gpu_id 0 \
#                 --do_unlearning \
#                --specific_settings \
#                --sgld \
#                --gamma .9 \
#                --dist cauchy \
#                --name  'unlearn_CFP_CLINIC_specific_settings' \
#                 --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
#                 --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_class_0_40.pth' 


# python main.py --forget_class 2 \
#                --data_name oculoplastic \
#                --model_name resnet \
#                --dataset_dir ./oculoplastic \
#                --gpu_id 0 \
#                --do_unlearning \
#                --specific_settings \
#                --sgld \
#                --gamma .9 \
#                --dist cauchy \
#                 --name 'unlearn_class_TED_sota_sgld_closest_points_unlearning_INVERSE_scale' \
#                 --original_model 'model_checkpoints/resnet_oculoplastic/resnetoculoplastic_original_model_30_3_class.pth' \
#                 --retrain_model  'model_checkpoints/resnet_oculoplastic/resnetoculoplastic_retrain_model_class_2_30_3_class.pth' 


# python main.py --custom_unlearn \
#                 --oculoplastics \
#                --data_name oculoplastic \
#                --model_name resnet \
#                --dataset_dir ./oculoplastic \
#                 --gpu_id 2 \
#                --do_unlearning \
#                --specific_settings \
#                --sgld \
#                --gamma .9 \
#                --dist cauchy \
#                --name 'unlearn_oculoplastic_vert_pf_specific_settings' \
#                 --original_model 'model_checkpoints/resnet_oculoplastic/resnetoculoplastic_original_model_30_3_class.pth' \
#                 --retrain_model  'model_checkpoints/resnet_oculoplastic/resnetoculoplastic_retrain_model_vert_pf_30_3_class.pth' 




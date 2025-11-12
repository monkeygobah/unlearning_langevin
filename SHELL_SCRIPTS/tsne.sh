
# python main.py --custom_unlearn \
#                 --to_forget 'Cirrus 800 FA' \
#                --data_name fundus_3_class \
#                --model_name resnet \
#                --dataset_dir ./data/fundus_big \
#                --gpu_id 0 \
#                --batch_size 1\
#                --tsne_embeddings \
#                --embeddings_name 'camera_unlearn' \
#                --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
#                --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_40_16_bs_class_cirrus.pth' \
#                --unlearn_model  'model_checkpoints/resnet_clinical_fundus/SPECIFOC_fundus_3_class_Cirrus_800_FA_lamnda_0.1_gamma_0_normal_resnet.pth'



python main.py --custom_unlearn \
                --oculoplastics \
                 --data_name oculoplastic \
               --model_name resnet \
               --dataset_dir ./data/oculoplastic \
               --gpu_id 0 \
               --batch_size 1\
               --tsne_embeddings \
               --embeddings_name 'oculoplastics' \
                --original_model 'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_original_model_30.pth' \
                --retrain_model  'model_checkpoints/resnet_oculoplastic/resnet_oculoplastic_retrain_retrain_OCULOPLASTIC_CLASS_VPF_29.pth' \
               --unlearn_model  'model_checkpoints/resnet_oculoplastic/unlearn_OCULOPLASTIC_CLASS_VPF_LAMDA_001.pth'

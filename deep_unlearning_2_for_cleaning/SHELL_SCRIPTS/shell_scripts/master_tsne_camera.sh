

python main.py --custom_unlearn \
                --to_forget 'Cirrus 800 FA' \
               --data_name fundus_3_class \
               --model_name resnet \
               --dataset_dir ./fundus_big \
               --gpu_id 0 \
               --batch_size 1\
               --tsne_embeddings \
               --embeddings_name 'camera_unlearn' \
               --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
               --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_40_16_bs_class_cirrus.pth' \
               --unlearn_model  'model_checkpoints/resnet_clinical_fundus/SPECIFOC_fundus_3_class_Cirrus_800_FA_lamnda_0.1_gamma_0_normal_resnet.pth'

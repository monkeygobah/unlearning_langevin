python main.py  --data_name dr_grade \
                --forget_class 4 \
                --model_name resnet \
                --dataset_dir ./data/dr_grading/classified_images_li \
                --lr 0.0001 \
                --epoch 50  \
                --batch_size 16  \
                --train \
                --gpu_id 0 \
                --name 'unlearn_DR_GRADE_sota_sgld_closest_points_unlearning_NO_scale_DELETE' \
    



# python main.py --custom_unlearn \
#                 --to_forget 'Cirrus 800 FA' \
#                --data_name fundus_3_class \
#                --model_name resnet \
#                --lr 0.001 \
#                --epoch 40  \
#                --dataset_dir ./fundus_big \
#                --batch_size 16 \
#                --gpu_id 0 \
#                --sgld \
#                --closest_points \
#                --run_sota \
#                --do_unlearning \
#                --name 'unlearn_CAMERA_CIRRUS_sota_sgld_closest_points_unlearning_NO_scale' \
#                 --original_model 'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_original_model_40.pth' \
#                 --retrain_model  'model_checkpoints/resnet_clinical_fundus/resnetfundus_3_class_retrain_model_40_16_bs_class_cirrus.pth' \

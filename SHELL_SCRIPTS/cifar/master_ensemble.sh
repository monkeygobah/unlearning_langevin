python main.py --forget_class 0 \
               --data_name cifar10 \
               --model_name AllCNN \
                --gpu_id 0 \
                --batch_size 64 \
                --name 'unlearn_cifar_class_0_ENSEMBLE' \
                --original_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_original_model_25.pth' \
                --retrain_model 'model_checkpoints/allcnn_cifar10/AllCNNcifar10_retrain_model_class_0_25.pth' \
                --good_forget 'model_checkpoints/ensemble_models/SGLD_cifar10_0_gamma:_0_normal_lamda:0.0_remain_reg0.0001_no_preproc_good_F.pth'\
                --good_remain 'model_checkpoints/ensemble_models/SGLD_cifar10_0_gamma:_0_normal_lamda:0.0_remain_reg0.0001_preprocess_good_R.pth'\
                --ensemble 


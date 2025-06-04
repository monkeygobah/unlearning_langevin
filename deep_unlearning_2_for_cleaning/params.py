import argparse


def get_parameters():
    parser = argparse.ArgumentParser("Boundary Unlearning")

    # this is always boundary shrink
    parser.add_argument('--method', type=str, default='boundary_shrink',
                        choices=['boundary_shrink', 'boundary_expanding'], help='unlearning method')
    
    
    # which dataset to use
    # TODO figure out why VIT is here and remove if not going to break anything
    parser.add_argument('--data_name', type=str, default='cifar10', choices=['mnist', 'cifar10', 'resnet', 'vit', 'open_source', 'fundus_3_class', 'oct_4_class', 'oculoplastic',\
     'dr_grade', 'mri', 'ultrasound', 'cxr', 'svhn', 'fashionmnist', 'medmnist'],
                        help='dataset, mnist or cifar10')
    
    # Which model to use
    #TODO if data_name is cifar, then have to use AllCNN
    parser.add_argument('--model_name', type=str, default='resnet', choices=['MNISTNet', 'AllCNN', 'resnet', 'vit'], help='model name')
    
    # Model settings
    parser.add_argument('--optim_name', type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer name')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=50, help='training epoch')
    
    
    parser.add_argument('--dataset_dir', type=str, default='./data', help='dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./model_checkpoints',
                        help='checkpoints directory')
    
    parser.add_argument('--do_unlearning', action='store_true', help='Only unlearning')

    # set train, retrain, or unlearn. Do in separate steps for easier experimentation
    parser.add_argument('--retrain_only', action='store_true', help='retrain dropping a new class')
    parser.add_argument('--train', action='store_true', help='Train model from scratch')

    
    # args for removal
    parser.add_argument('--sgld', action='store_true', help='SGLD shrinkage')
    parser.add_argument('--closest_points', action='store_true', help='do a closest point experiment') 
    parser.add_argument('--run_sota', action='store_true', help='run sota method from chen et al')
    parser.add_argument('--specific_settings', action='store_true', help='run unlearning with specific setting ')


    # args for defining what to be unlearned
    parser.add_argument('--forget_class', type=int, default=2, help='forget class')
    parser.add_argument('--custom_unlearn', action='store_true', help='Whether or not to unlearn based on metadata')
    parser.add_argument('--to_forget', type=str, default='Cirrus 800 FA', help='Feature to unlearn from dataset')
    parser.add_argument('--oculoplastics', action='store_true', help='Use oculoplastic dataset')

    # training params
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--extra_exp', action='store_true')


    # model paths for unlearning
    parser.add_argument('--original_model', type=str, help='path to original model')
    parser.add_argument('--retrain_model', type=str, help='path to retrain model')
    
    
    # algorithm args
    parser.add_argument('--use_linfpgd', action='store_true', help='use linfpgd- only for unlearning ViT')
    parser.add_argument('--scaling', type=str, default='None', choices=['None', 'inverse', 'exponential_decay', 'linear_decay'], help='lambda scaling method')

    parser.add_argument('--gpu_id', type=int,default = 0, help='which GPU to use') 
    parser.add_argument('--gamma', type=float, default = 0)
    parser.add_argument('--lamda', type=float, default = 0)
    parser.add_argument('--dist', type=str)
    parser.add_argument('--relearn', action='store_true', help='run RElearning experiment on unlearned model ')
    parser.add_argument('--name', type=str, default = 'placeholder')

    #params for embedding experiments
    parser.add_argument('--tsne_embeddings', action='store_true', help='Only unlearning')
    parser.add_argument('--unlearn_model', type=str, help='path to unlearn model')
    parser.add_argument('--embeddings_name', type=str, help='path to unlearn model')


    parser.add_argument('--use_logits', action='store_true', help='toggle whether or not to logits or argmax of adv sample')
    parser.add_argument('--remain_reg', type=float, help='contribution of remain loss to add to total loss')
    parser.add_argument('--logit_preprocess', action='store_true', help='toggle whether or not to preprocess logits')


    parser.add_argument('--good_forget', type=str, help='path to unlearn model with good forget acc')
    parser.add_argument('--good_remain', type=str, help='path to unlearn model with good remain acc')
    parser.add_argument('--ensemble', action='store_true', help='toggle whether or not to do enseble experiments')



    parser.add_argument('--percent_to_forget', type=float,  default =1)
    parser.add_argument('--selective_unlearn', action='store_true', help='toggle whether or not to do selective unlearning experiments')

    
    args = parser.parse_args()


    if args.do_unlearning is False:
        if any([args.sgld, args.closest_points, args.run_sota, args.specific_settings]):
            raise ValueError("SGLD, closest_points, run_sota, and specific_setting can only be set if --do_unlearning is true")


    if args.do_unlearning:
        if not args.original_model:
            raise ValueError("If --unlearn_only is true, --original_model must be defined")
        if not args.retrain_model:
            raise ValueError("If --unlearn_only is true, --retrain_model must be defined")

    if args.data_name == 'cifar10' and args.model_name != 'AllCNN':
        raise ValueError("If --data_name is 'cifar10', --model_name must be 'AllCNN'")


    return args
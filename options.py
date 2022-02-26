import argparse, os, utils

class Options():
    """This class defines options for training, testing and prediction"""

    def __init__(self):
        self.initialized = False
    
    def initialize(self, parser):
        self.initialized = True

        #######################################################################
        ### Options to be used when doing both hyper-parameter and training ###
        #######################################################################

        # Data options
        parser.add_argument('--dataset_contains', type=str, default='train_and_test', required=True, choices=["train_and_test", "train"], help='Mention whether the dataset contains training samples only or both training and test samples. Combined dataset is used for cross-validation.')
        parser.add_argument('--input_channels', type=int, default=1, required=False, help='Number of channels in the input data.')
        parser.add_argument('--output_channels', type=int, default=1, required=False, help='Number of channels in the output label.')
        parser.add_argument('--voxel_spacing', type=float, nargs=3, default=(0.6, 0.6, 0.6), required=False, help='New voxel spacing for data.')
        parser.add_argument('--img_mod', type=str, default='MR', choices=["CT", "MR"], required=True, help='Image modality.')

        # Data handling and processing
        parser.add_argument('--n_batches', type=int, default=2, required=False, help='Number of batches sampled for training.')
        parser.add_argument('--n_workers', type=int, default=5, required=False, help='Number of workers for data loading.')
        parser.add_argument('--p_foreground', type=float, default=0.6, required=False, help='Probability of sampling patch with prostate included.')
        parser.add_argument('--patch_shape', type=int, nargs=3, default=(160, 160, 64), required=False, help='Patch shape for random patch selection. Usage: "--patch_shape 128 128 128"')
        parser.add_argument('--margin', type=int, default=2, required=False, help='Margin for patch sampling so that less information is lost during augmentation. Must be even number.')
        parser.add_argument('--no_augmentation', action='store_true', required=False, help='Turn off data augmentation during training.')
        parser.add_argument('--normalize', type=str, default='local', required=False, help='Choose type of normalization {None, local, global, -11}.')
        parser.add_argument('--no_clip', action='store_true', required=False, help='If set, data will not be clipped to 5 and 95 percentiles.')
        parser.add_argument('--switch', action='store_true', required=False, help='If set, perform normalization first and then augmentation. If not set, perform augmentation first and then normalization.')
        parser.add_argument('--fold', type=int, default=-1, required=False, choices = [-1,0,1,2,3,4], help='Select the fold index for 5-fold crosss validation. If fold == -1, then no cross-validation, but randomly splits the train/test:80/20.')
        parser.add_argument('--store_loaded_data', action='store_true', required=False, help='Store the loaded data in main memory? This will take way more memory, but way less computing time.')
        parser.add_argument('--no_shuffle', action='store_true', required=False, help='If set, training and validation data will not be shuffled')

        # Model options
        parser.add_argument('--model_name', type=str, default='unet', required=False, choices=["unet", "attention_unet", "inception_unet", "deeper_unet_ump", "unet_wmp", "vnet", "shallow_vnet", "dense_voxel_net", "oocs_unet_k3", "unet_pp", "unet++", "oocs_vnet_k3", "oocs_vnet_k5", "oocs_unet_k5", "oocs_attention_unet_k5", "oocs_attention_unet_k3"], help='Select model from the list.')
        parser.add_argument('--n_kernels', type=int, default=32, required=False, help='Number of kernels for the fist Conv layer of Discriminator. Following kernels are multiplicatives.')
        parser.add_argument('--eps', type=float, default=1e-07, required=False, help='Epsilon for Adam solver.')
        parser.add_argument('--weight_decay', type=float, default=0.0, required=False, help='Weight decay for Adam solver.')
        parser.add_argument('--beta2', type=float, default=0.999, required=False, help='Beta2 for Adam solver.')
        parser.add_argument('--loss_fn', type=str, default='bce_dice_loss', required=False, choices=["binary_cross_entropy", "dice", "bce_dice_loss"], help='Choose loss function.')
        parser.add_argument('--optimizer', type=str, default='adam', required=False, help='Choose optimizer {adam, adamax}.')

        # Paths
        parser.add_argument('--results_path', type=str, default='results/train/', required=False, help='Path to store the results of the training and hyper-parameter search.')
        parser.add_argument('--data_root', type=str, default='data/train_and_test', required=True, help='Path to data samples.')
        
        # Hardware device selection
        parser.add_argument('--device_id', type=int, default=0, required=False, help='Use the different GPU(device) numbers available. Example: If two Cuda devices are available, options: 0 or 1')
        
        ########################################################################
        ### Additional options to be used during hyper-parameter search only ###
        ########################################################################
        parser.add_argument('--beta1_min', type=float, default=0.1, required=False, help='Minimum Beta1 for Adam solver.')
        parser.add_argument('--beta1_max', type=float, default=0.9, required=False, help='Maximum Beta1 for Adam solver.')
        parser.add_argument('--lr_min', type=float, default=0.000001, required=False, help="Minimum Learning rate for model's Adam solver.")
        parser.add_argument('--lr_max', type=float, default=0.1, required=False, help="Maximum Learning rate for model's Adam solver.")
        parser.add_argument('--dropout_rate_min', type=float, default=0.0, required=False, help='Minimum Dropout rate for model.')
        parser.add_argument('--dropout_rate_max', type=float, default=0.7, required=False, help='Maximum Dropout rate for model.')
        parser.add_argument('--run_id', type=int, default=0, required=False, help='Run ID for hpbandster.')
        parser.add_argument('--port', type=int, default=0, required=False, help='Port for hpbandster.')
        parser.add_argument('--min_budget', type=int, default=100, required=False, help='Minimum number of epochs per configuration.')
        parser.add_argument('--max_budget', type=int, default=500, required=False, help='Maximum number of epochs per configuration.')
        parser.add_argument('--n_iterations', type=int, default=7, required=False, help='Number of iterations for hpbandster.')
        parser.add_argument('--previous_search_path', type=str, default='None', required=False, help='Use this with hyperparam_optimization.py file. If not None, the parameter search will resume from the previous run directory.')

        #######################################################################
        ##### Additional options to be used when doing only training ##########
        #######################################################################
        parser.add_argument('--training_epochs', type=int, default=500, required=False, help='Number of training epochs.')
        parser.add_argument('--starting_epoch', type=int, default=0, required=False, help='Epoch to continue training from.')
        parser.add_argument('--resume_train', action='store_true', required=False, help='Use this with train.py file. If set, the training will resume from the last saved epoch.')
        parser.add_argument('--save_freq', type=int, default=0, required=False, help='Saves all validation samples in the predictions folder after certain epcohs have elapsed. save_freq = 0: no saving, save_freq > 0: saving')
        parser.add_argument('--beta1', type=float, default=0.9, required=False, help='Beta1 for Adam solver.')
        parser.add_argument('--lr', type=float, default=1e-4, required=False, help="Learning rate for model's Adam solver.")
        parser.add_argument('--dropout_rate', type=float, default=0.5, required=False, help='Dropout rate for model.')
        
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        msg = ''
        msg += '--------------- Options -----------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            msg += '{:>20}: {:<30}{}\n'.format(str(k), str(v), comment)
        msg += '----------------- End -------------------'
        print(msg)
        return msg        
      
    def save_options(self, msg, opt):
        """
        Write options into a file 'training_options.txt'
        """
        # Check if results folder exists and if so, ask user if really want to continue.
        utils.overwrite_request(opt.results_path)
        utils.create_folder(opt.results_path)
        #utils.create_folder(os.path.join(opt.results_path, 'predictions'))

        # Write options into a file.
        with open(opt.results_path + '/training_options.txt', 'w') as f:
            f.write(msg)

    def parse(self):
        opt = self.gather_options()
        self.opt = opt
        msg = self.print_options(opt)
        self.save_options(msg, opt)
        return self.opt

import argparse

parser = argparse.ArgumentParser(description='Super Resolution')

# Hardware specifications
parser.add_argument('--cuda', action='store_true', 
                    help='use cuda?')
parser.add_argument('--description', type=str, default='test aug data', 
                    help='log operations')
parser.add_argument('--seed', type=int, default=123, 
                    help='random seed to use. Default=123')
parser.add_argument('--threads', type=int, default=3, 
                    help='number of threads for data loader to use')

# Data specifications
parser.add_argument('--upscale', default=2, 
                    help="super resolution upscale factor")
parser.add_argument('--dir_datasets', type=str, default='../datasets',
                    help='dataset directory')
parser.add_argument('--data_train', type=str, default='B100', 
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='B100', 
                    help='test dataset name')
parser.add_argument('--crop_size', type=int, default=256, 
                    help='train image crop size')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--aug', action='store_true',
                    help='augmention of training data')


# Model specifications

parser.add_argument('--model', type=str, default='EDSR',
                    choices=('EDSR', 'DDBPN'),
                    help='model to use (ESSR | DDBPN)')              
parser.add_argument('--n_resblocks', type=int, default=3, 
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64, 
                    help='number of feature maps')
parser.add_argument('--train_batch', type=int, default=25, 
                    help='training batch size')
parser.add_argument('--epochs', type=int, default=10, 
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=2e-4, 
                    help='learning rate. Default=2e-4')
parser.add_argument('--init_weight', action='store_true', 
                    help='initalize weights?')
parser.add_argument('--loss', default='L1', 
                    choices=('MSE', 'L1', 'Robust', 'Perceptual'), 
                    help='loss function to use (MSE | L1 | Robust | Perceptual)')

# Training specifications
parser.add_argument('--n_train', type=int, default=800,
                    help='number of training data of DIV2K dataset')
parser.add_argument('--reset', action='store_true', 
                    help='reset the training')
parser.add_argument('--resume', type=int, default=-1, 
                    help='load the model from the specified epoch')
parser.add_argument('--log_file', type=str, default='logs', 
                    help='log file name')
parser.add_argument('--print_freq', type=int, default=3,
                    help='frequency of showing training results on console')
parser.add_argument('--test_freq', type=int, default=5,
                    help='frequency of testing testset on console')

# Testing specifications
parser.add_argument('--test', action='store_true', 
                    help='test the model with data_test')
parser.add_argument('--pre_train', type=str, default='.', 
                    help='load pre-trained model to test')
parser.add_argument('--save', action='store_true', 
                    help='save the model result')

args = parser.parse_args()
print(args)
args.upscale = list(map(lambda x: int(x), args.upscale.split('+')))


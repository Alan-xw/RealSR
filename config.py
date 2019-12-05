
# coding :utf-8
import argparse
# DATA
DIR_Train = '../Image_data/'
DIR_Test = '../Image_data/Test/'
DIR_DEMO = './Test_real_small/'
SCALE = 2
BATCH_SIZE = 16
INPUT_SIZE = 192
DATA_NAME ='Real'
DEMO_NAME = 'Real_16'
TEST_NAME = ['Real']

# TRAIN and TEST
TEST_ONLY = False
CPU = False
N_GPUs = 1
# LOSS = '1*VGG54+0.5*L1+0.001*RLSGAN+1*L1+0.01*TV'
LOSS = '1*L1'
PRECISION = 'single'
# MODEL
MODEL_NAME = 'RealSR'

parser = argparse.ArgumentParser(description="SuperResolution_V1")

# Model_1 EDSR
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')

parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')

parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')



# DATA configuration
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help="Number of images a batch")
parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                    help="Size of training input.")
parser.add_argument("--scale", type=int, default=SCALE,
                    help="Scale factor")
parser.add_argument("--dir-train", type=str, default=DIR_Train,
                    help="Dir of LR images")
parser.add_argument("--dir-test", type=str, default=DIR_Test,
                    help="Dir of HR images")
parser.add_argument("--dir-demo", type=str, default=DIR_DEMO,
                    help="Dir of HR images")
parser.add_argument("--data-name", type=str, default=DATA_NAME,
                    help="name of training dataset")
parser.add_argument("--demo-name", type=str, default=DEMO_NAME,
                    help="name of training dataset")
parser.add_argument("--test-name", type=list, default=TEST_NAME,
                    help="name of test dataset")
parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')


# Training and Testing configuration
parser.add_argument("--test-only", type=bool, default=False,
                    help="Test only or not")
parser.add_argument("--demo", type=bool, default=False,
                    help="Test only or not")
parser.add_argument("--cpu", type=bool, default=CPU,
                    help="use coy only")
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--loss', type=str, default=LOSS,
                    help='loss function configuration')
parser.add_argument('--self_ensemble', default=False,
                    help='use self-ensemble method for test')
parser.add_argument('--chop', default=False,
                    help='use chop method for test')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train')


parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
# Model configuration
parser.add_argument("--model-name", type=str, default=MODEL_NAME,
                    help="name of the model")

# Optimizer
parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
parser.add_argument('--decay', type=str, default='1000',
                    help='learning rate decay type') 
# 250-500-750-1000-1250-1500
# 375-625-875-1125-1375-1625
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Log specifications
parser.add_argument('--save', type=str, default='RealSR_modified_head',
                    help='file name to save')
parser.add_argument('--load', type=str, default='RealSR_modified_head',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=-1,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', default=True,
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')


args = parser.parse_args()



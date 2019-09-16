from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('./')

import six
import os
import os.path as osp
import math
import argparse


parser = argparse.ArgumentParser(description="Softmax loss classification")
# data
parser.add_argument('--run_on_remote', action='store_true', default=False,
                    help="run the code on remote or local.")


# from main args
parser.add_argument('--resize', type = int, default = 5)
parser.add_argument('--sharpness', type = int, default = 1)
parser.add_argument('--contrast', type = int, default = 3)
#parser.add_argument('--scale', type = str, default = None)
parser.add_argument('--psm', type = str, default = '6')
parser.add_argument('--model', type = str, default = 'wips')

#   aster
#parser.add_argument('-b', '--batch_size', type=int, default=128)
parser.add_argument('-j', '--workers', type=int, default=8)
parser.add_argument('--height', type=int, default=64,
                    help="input height, default: 256 for resnet*, ""64 for inception")
parser.add_argument('--width', type=int, default=256,
                    help="input width, default: 128 for resnet*, ""256 for inception")
parser.add_argument('--keep_ratio', action='store_true', default=False,
                    help='length fixed or lenghth variable.')
parser.add_argument('--voc_type', type=str, default='ALLCASES_SYMBOLS',
                    choices=['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS'])
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--with_lstm', action='store_true', default=False,
                    help='whether append lstm after cnn in the encoder part.')
parser.add_argument('--decoder_sdim', type=int, default=512,
                    help="the dim of hidden layer in decoder.")
parser.add_argument('--attDim', type=int, default=512,
                    help="the dim for attention.")

# optimizer
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0) # the model maybe under-fitting, 0.0 gives much better results.
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--loss_weights', nargs='+', type=float, default=[1,1,1])                    

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--cuda', default=True, type=bool,
                    help='whether use cuda support.')

parser.add_argument('--beam_width', type=int, default=5)


parser.add_argument('--n_epochs', type = int, default = 10)
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--n_group', type = int, default = 1)
parser.add_argument('--rec_num_classes', type = int, default = 26)

#   iter params
parser.add_argument('--max_len', type = int, default = 18) # for final version
parser.add_argument('--epochs', type = int, default = 6)
parser.add_argument('--lr', type = float, default = 0.1)

parser.add_argument('--image_format', type = str, default = 'pil')
parser.add_argument('--eos', type = int, default = 94)


parser.add_argument('--eval', type = bool, default = True)
parser.add_argument('--save_preds', type = bool, default = False)






def get_args(sys_args):
  global_args = parser.parse_args(sys_args)
  return global_args

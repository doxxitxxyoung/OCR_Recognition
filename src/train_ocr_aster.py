#!usr/bin/python
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from PIL import ImageEnhance
import cv2
#from ocr import get_ocr
import glob

import argparse
import os
import sys

#from file_list_final import file_train_list, file_val_list


from sklearn.model_selection import KFold

#from lib.models.model_builder import ModelBuilder
from lib.models.resnet_aster import ResNet_ASTER
from lib.models.attention_recognition_head import AttentionRecognitionHead
from lib.loss.sequenceCrossEntropyLoss import SequenceCrossEntropyLoss
#from lib.models.tps_spatial_transformer import TPSSpatialTransformer
#from config import get_args

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, SubsetRandomSampler

from lib.datasets.dataset import AlignCollate, ResizeNormalize
from lib.utils.serialization import load_checkpoint, save_checkpoint

import pickle
import random

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

def get_data(filename, args):
    """
    @ input
    filename: ex. kr00001973962b1p-4
    
    @ output
    image, coordinates, labels
    """
    
    xmlfile = filename
    jpgfile = filename.replace(".xml",".jpg")
    
    doc = ET.parse(xmlfile)
    root = doc.getroot()
    object_dict = {}
    for x in root.findall('object'):
        coord = '/'.join([x.find('bndbox').find('xmin').text,
                x.find('bndbox').find('ymin').text,
                x.find('bndbox').find('xmax').text,
                x.find('bndbox').find('ymax').text])
        label = x.find('name').text
        object_dict[coord] = label

    if args.image_format == 'cv2':
        image = cv2.imread(jpgfile, cv2.IMREAD_COLOR)
    else:
        image = Image.open(jpgfile)

    coordinates = []
    labels = []
    for key in object_dict.keys():
        coord = tuple([int(x) for x in key.split('/')])
        coordinates.append(coord)
        labels.append(object_dict[key])
        
    return image, coordinates, labels

def get_files():
    """
    @output
    filenames
    """
    filenames = glob.glob("../data/*.xml")
    return filenames
    
def crop_image(image,coordinates, args, resample = Image.BICUBIC):
    """
    @Input
    image: an image (PIL)
    coordinates: a list of coordinates of text which should be cropped
    
    @Output
    cropped_images: a list of cropped images
    """

    #   PIL coordinate : (w0, h0, w1, h1)
    #   cv2 coordinate : [h0:h1, w0:w1]

    cropped_images = []
    for i,coordinate in enumerate(coordinates):
        if args.image_format == 'cv2':
            cropped_image = image[coordinate[1]:coordinate[3], coordinate[0]:coordinate[2]]
            
        else:
            cropped_image = image.crop(coordinate)

            # resize
            h, w = cropped_image.size

            # ratio = 1.5
            cropped_image = cropped_image.resize((int(h*args.resize), int(w*args.resize)),
                                                    resample= resample)
            sharpness_enhancer = ImageEnhance.Sharpness(cropped_image)
            cropped_image = sharpness_enhancer.enhance(args.sharpness)
            contrast_enhancer = ImageEnhance.Contrast(cropped_image)
            cropped_image = contrast_enhancer.enhance(args.contrast)
            
        cropped_images.append(cropped_image)
    
    return cropped_images

def Create_char_dict(args):
    """
    Returns Charactor to indexes / indexes to character dictionaries
    """
    import string
    voc = list(string.printable[:-6])
    voc.append('EOS')
    voc.append('PADDING')
    voc.append('UNKNOWN')

    char2id_dict = dict(zip(voc, range(len(voc))))
    id2char_dict = dict(zip(range(len(voc)), voc))

    return args, char2id_dict, id2char_dict



def Create_data_list_byfolder(args, char2id, id2char, file_list):
    """
    inputs
    file_list : list of file names
    char2id : dictionary mapping charactors to indexes
    id2char : dictionary mapping indexes to charactors

    output
    input_list : list of training samples. each samples are dictionaries with keys : 'images', 'rec_targets', 'rec_lengths',

    """

    #   input path for individual data list
    filenames = [x+'.xml' for x in file_list]
    print('number of files : '+str(len(filenames)))


    #   get train-test datalist (pil image)

    input_list = []

    for filename in filenames:
        try:

            image, coordinates, labels = get_data(filename, args)

            cropped_images = crop_image(image, coordinates, args, resample = Image.BICUBIC)

            for i, crop in enumerate(cropped_images):
                #   convert to cv2 format

                if args.image_format == 'cv2':
                    crop = crop
                else:
                    crop_cv2 = np.asarray(crop, dtype = np.float)

                ## fill with the padding token
                label = np.full((args.max_len,), char2id['PADDING'], dtype=np.int)
                label_list = []
                for char in labels[i]:
                    if char in char2id:
                        label_list.append(char2id[char])
                    else:
                        ## add the unknown token
                        print('{0} is out of vocabulary.'.format(char))
                        label_list.append(char2id['UNKNOWN'])

                ## add a stop token
                label_list = label_list + [char2id['EOS']]

                label[:len(label_list)] = np.array(label_list)


                
                # label length
                label_len = len(label)

                input_list.append({'images' : crop, 
                                    'rec_targets' : label,
                                    'rec_lengths' : label_len})
        except:
            pass

    return input_list

#   ver 2
def main_aster(folder_name):
    """
    @Input
    folder_name : name of the folder where training data are stored.
     
    @Output
    trained parameters are stored in 'params' folder
    """

    #   arguments are stored in pred_params.py
    from pred_params import Get_ocr_args
    args = Get_ocr_args()

    print('Evaluation : '+str(args.eval))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    if args.cuda:
        print('using cuda.')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print('using cpu.')
        torch.set_default_tensor_type('torch.FloatTensor')


    #   Create Character dict & max seq len
    args, char2id_dict , id2char_dict= Create_char_dict(args)

    print(id2char_dict)
    rec_num_classes = len(id2char_dict)

    #   Get rec num classes / max len
    print('max len : '+str(args.max_len))
    

    #   Get file list for train set
    filenames = glob.glob('./data/' + folder_name + '/*/*.xml')
    filenames = [x[:-4] for x in filenames]
    print('file len : '+str(len(filenames)))

    #   files are not splitted into train/valid set.
    train_list = Create_data_list_byfolder(args, char2id_dict, id2char_dict, filenames)
    
    encoder = ResNet_ASTER(with_lstm = True, n_group = args.n_group, use_cuda = args.cuda)

    encoder_out_planes = encoder.out_planes

    decoder = AttentionRecognitionHead(num_classes = rec_num_classes,
                                        in_planes = encoder_out_planes,
                                        sDim = args.decoder_sdim,
                                        attDim = args.attDim,
                                        max_len_labels = args.max_len,
                                        use_cuda = args.cuda)

    #   Load pretrained weights
    if not args.eval:
        if args.use_pretrained:
            #   use pretrained model
            pretrain_path = './data/demo.pth.tar'
            if args.cuda:
                pretrained_dict = torch.load(pretrain_path)['state_dict']
            else:
                pretrained_dict = torch.load(pretrain_path, map_location='cpu')['state_dict']
                
            encoder_dict = {}
            decoder_dict = {}
            for i, x in enumerate(pretrained_dict.keys()):
                if 'encoder' in x:
                    encoder_dict['.'.join(x.split('.')[1:])] = pretrained_dict[x]
                elif 'decoder' in x:
                    decoder_dict['.'.join(x.split('.')[1:])] = pretrained_dict[x]
            encoder.load_state_dict(encoder_dict)
            decoder.load_state_dict(decoder_dict)
            print('pretrained model loaded')

        else:
            #   init model parameters
            def init_weights(m):
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_uniform(m.weight)
                    #m.bias.data.fill_(0.01)

            encoder.apply(init_weights)
            decoder.apply(init_weights)
            print('Random weight initialized!')

    else:
        #   loading parameters for inference
        if args.cuda:
            encoder.load_state_dict(torch.load('params/encoder_final'))
            decoder.load_state_dict(torch.load('params/decoder_final'))
        else:
            encoder.load_state_dict(torch.load('params/encoder_final', map_location=torch.device('cpu')))
            decoder.load_state_dict(torch.load('params/decoder_final', map_location=torch.device('cpu')))
        print('fine-tuned model loaded')

    #   Training Phase

    rec_crit = SequenceCrossEntropyLoss()

    if (args.cuda == True) & torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    encoder.to(device)
    decoder.to(device)

    param_groups = encoder.parameters()
    param_groups = filter(lambda p: p.requires_grad, param_groups)
    optimizer = torch.optim.Adadelta(param_groups, lr = args.lr, weight_decay = args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [4,5], gamma = 0.1)

    train_loader = DataLoader(train_list, 
                            batch_size = args.batch_size,
                            shuffle = False,
                            collate_fn = AlignCollate(
                                imgH = args.height, imgW = args.width, keep_ratio = True)
                            )

    for epoch in range(args.n_epochs):
        for batch_idx, batch in enumerate(train_loader):

            x, rec_targets, rec_lengths = batch[0], batch[1], batch[2]

            x = x.to(device)
            encoder_feats = encoder(x) # bs x w x C
            rec_pred = decoder([encoder_feats, rec_targets, rec_lengths])
            loss_rec = rec_crit(rec_pred, rec_targets, rec_lengths)
            
            if batch_idx == 0:
                print('train Loss : '+str(loss_rec))
                rec_pred_idx = np.argmax(rec_pred.detach().cpu().numpy(), axis = -1)
                print(rec_pred[:3])
                print(rec_pred_idx[:5])

            optimizer.zero_grad()
            loss_rec.backward()
            optimizer.step()

    #   Training phase ends

    #   this is where trained model parameters are saved

    torch.save(encoder.state_dict(), 'params/encoder_final')
    torch.save(decoder.state_dict(), 'params/decoder_final')

if __name__ == "__main__":

    
    Folder_name = 'dataset_final'   #   This is the folder where training data are stored in 'data' folder. 
    main_aster(Folder_name)



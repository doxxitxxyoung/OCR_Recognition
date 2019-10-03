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

from file_list_final import file_train_list, file_val_list


from sklearn.model_selection import KFold

#from lib.models.model_builder import ModelBuilder
from lib.models.resnet_aster import ResNet_ASTER
from lib.models.attention_recognition_head import AttentionRecognitionHead
from lib.loss.sequenceCrossEntropyLoss import SequenceCrossEntropyLoss
#from config import get_args

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, SubsetRandomSampler

from lib.datasets.dataset import AlignCollate, ResizeNormalize
from lib.utils.serialization import load_checkpoint, save_checkpoint

import pickle

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
#            coordinate = [coordinate[0],coordinate[2], coordinate[1],coordinate[3]]
#            cropped_image = image[coordinate[0]:coordinate[2], coordinate[1]:coordinate[3]]
            cropped_image = image[coordinate[1]:coordinate[3], coordinate[0]:coordinate[2]]
            
        else:
            cropped_image = image.crop(coordinate)
            # print("cropped_image.size=",cropped_image.size)

            # resize
            h, w = cropped_image.size

            # ratio = 1.5
            cropped_image = cropped_image.resize((int(h*args.resize), int(w*args.resize)),
                                                    resample= resample)
            # print("cropped_image.size=",cropped_image.size)
            sharpness_enhancer = ImageEnhance.Sharpness(cropped_image)
            cropped_image = sharpness_enhancer.enhance(args.sharpness)
            contrast_enhancer = ImageEnhance.Contrast(cropped_image)
            cropped_image = contrast_enhancer.enhance(args.contrast)
            
            # if i == 0:
            #     cropped_image.show()

        cropped_images.append(cropped_image)
    
    return cropped_images

def Create_char_dict(args):
    import string
    voc = list(string.printable[:-6])
    voc.append('EOS')
    voc.append('PADDING')
    voc.append('UNKNOWN')

    char2id_dict = dict(zip(voc, range(len(voc))))
    id2char_dict = dict(zip(range(len(voc)), voc))

    return args, char2id_dict, id2char_dict


def Create_data_list(args, char2id, train):
    #   input path for individual data list
    if train == True:
        filenames = [x+'.xml' for x in file_train_list]
        print('train file number : '+str(len(filenames)))
    else:
        filenames = [x+'.xml' for x in file_val_list]
        print('test file number : '+str(len(filenames)))


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

    return input_list, args

#   ver 2
#def main_aster(args):
def main_aster():

#    from config import get_args
#    args = get_args(sys.argv[1:])

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

#    args.cuda = True and torch.cuda.is_available()

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
    
    #   Create data list
    train_list, args = Create_data_list(args, char2id_dict, True)
    test_list, args = Create_data_list(args, char2id_dict, False)



    encoder = ResNet_ASTER(with_lstm = True, n_group = args.n_group)

    encoder_out_planes = encoder.out_planes

    decoder = AttentionRecognitionHead(num_classes = rec_num_classes,
                                        in_planes = encoder_out_planes,
                                        sDim = args.decoder_sdim,
                                        attDim = args.attDim,
                                        max_len_labels = args.max_len)

    #   Load pretrained weights
    if not args.eval:
        pretrain_path = './data/demo.pth.tar'
        pretrained_dict = torch.load(pretrain_path)['state_dict']
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
#        encoder.load_state_dict(torch.load('../params/encoder_final'))
#        decoder.load_state_dict(torch.load('../params/decoder_final'))
        encoder.load_state_dict(torch.load('params/encoder_final'))
        decoder.load_state_dict(torch.load('params/decoder_final'))
        print('fine-tuned model loaded')


    rec_crit = SequenceCrossEntropyLoss()

    if args.cuda == True:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    encoder.to(device)
    decoder.to(device)

#    param_groups = model.parameters()
    param_groups = encoder.parameters()
    param_groups = filter(lambda p: p.requires_grad, param_groups)
    optimizer = torch.optim.Adadelta(param_groups, lr = args.lr, weight_decay = args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [4,5], gamma = 0.1)

    test_proba = []
    test_pred = []
    test_label = []
    test_image = []

    train_loader = DataLoader(train_list, 
                            batch_size = args.batch_size,
                            shuffle = False,
                            collate_fn = AlignCollate(
                                imgH = args.height, imgW = args.width, keep_ratio = True)
                            )
    test_loader = DataLoader(test_list, 
                            batch_size = args.batch_size,
                            shuffle = False,
                            collate_fn = AlignCollate(
                                imgH = args.height, imgW = args.width, keep_ratio = True)
                            )

    if not args.eval:
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

        if args.cuda:
            torch.save(encoder.state_dict(), 'params/encoder_final')
            torch.save(decoder.state_dict(), 'params/decoder_final')
        else:
            torch.save(encoder.state_dict(), 'params/encoder_final_cpu')
            torch.save(decoder.state_dict(), 'params/decoder_final_cpu')


    
    for batch_idx, batch in enumerate(test_loader):

        x, rec_targets, rec_lengths = batch[0], batch[1], batch[2]

        encoder_feats = encoder(x)
        rec_pred, rec_pred_scores = decoder.beam_search(encoder_feats,\
                                                args.beam_width, args.eos)

        rec_pred = rec_pred.detach().cpu().numpy()
        rec_targets = rec_targets.numpy()
        print('predictions')
        print(rec_pred[:5])
        print('label')
        print(rec_targets[:5])
        test_proba.extend(rec_pred_scores)
        test_pred.extend(rec_pred)
        test_label.extend(rec_targets)
        test_image.extend(x.detach().cpu().numpy())

        hit = 0
        miss = 0
        try:
            for i, x in enumerate(rec_pred):
                if rec_pred[i] == rec_targets[i]:
                    hit += 1
                else:
                    miss += 1

            accuracy = hit/(hit+miss)
            print("batch accuracy=",accuracy)
        except:
            pass
           
    hit = 0
    miss = 0

    if args.save_preds == True:
        with open('aster_pred.pkl', 'wb') as f:
            pickle.dump([test_label, test_pred, test_proba, char2id_dict, id2char_dict, test_image], f)

    def get_score(test_label, test_pred):
        total_n = 0
        true_n = 0
        eos = 94
        for i, x in enumerate(test_label):
            total_n += 1
            eos_idx = 0
            for j, y in enumerate(x):
                if y != eos:
                    eos_idx += 1
                else:
                    break
            label = x[:eos_idx]
            pred = test_pred[i][:eos_idx]
            if np.array_equal(label, pred):
                true_n += 1
        print('Accuracy')
        print(true_n/total_n)

    get_score(test_label, test_pred)

def get_data_pred(filename, args):
    """
    @ input
    filename: ex. kr00001973962b1p-4
    
    @ output
    image
    """
    
#        xmlfile = filename
    jpgfile = filename.replace(".xml",".jpg")
    
    if args.image_format == 'cv2':
        image = cv2.imread(jpgfile, cv2.IMREAD_COLOR)
    else:
        image = Image.open(jpgfile)

    return image


class Pred_Aster():
    def __init__(self):
        
        from config import get_args
        args = get_args(sys.argv[1:])

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        args.cuda = True and torch.cuda.is_available()

        if args.cuda:
            print('using cuda.')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        #   Create Character dict & max seq len
        args, char2id_dict , id2char_dict= Create_char_dict(args)

        print(id2char_dict)
        rec_num_classes = len(id2char_dict)

        #   Get rec num classes / max len
        print('max len : '+str(args.max_len))


        #   init model

        encoder = ResNet_ASTER(with_lstm = True, n_group = args.n_group)

        encoder_out_planes = encoder.out_planes

        decoder = AttentionRecognitionHead(num_classes = rec_num_classes,
                                            in_planes = encoder_out_planes,
                                            sDim = args.decoder_sdim,
                                            attDim = args.attDim,
                                            max_len_labels = args.max_len)

        encoder.load_state_dict(torch.load('params/encoder_final'))
        decoder.load_state_dict(torch.load('params/decoder_final'))
        print('fine-tuned model loaded')


        device = torch.device('cuda')
        encoder.to(device)
        decoder.to(device)

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.args = args
        self.char2id_dict = char2id_dict
        self.id2char_dict = id2char_dict

    def idx2char(self, pred, id2char_dict, eos='EOS'):
        eos_idx = [x for x in id2char_dict.keys() if id2char_dict[x]==eos][0]
        pred = pred[:pred.tolist().index(eos_idx)]
        pred = [id2char_dict[x] for x in pred]
        pred_char = ''.join(pred)
        return pred_char


    def forward(self, image_path, coordinates):

        """
        @input
        image paths : One image path without '.xml' or '.png'
        coordinates: A List of coordinates

        @output : A List of characters
        """

        args = self.args
        encoder = self.encoder
        decoder = self.decoder
        device = self.device

        image = get_data_pred(image_path, args)

        cropped_images = crop_image(image,coordinates, args, resample = Image.BICUBIC) # list of imgs

        cropped_images = [{'images' : x, 'rec_targets' : 0, 'rec_lengths' : 0} 
                           for x in cropped_images]

        #   data loader
        test_pred = []
        test_image = []

        test_loader = DataLoader(cropped_images, 
                                batch_size = args.batch_size,
                                shuffle = False,
                                collate_fn = AlignCollate(
                                    imgH = args.height, imgW = args.width, keep_ratio = True)
                                )

        for batch_idx, batch in enumerate(test_loader):

            x = batch[0].to(device)

            encoder_feats = self.encoder(x)
            rec_pred, rec_pred_scores = decoder.beam_search(encoder_feats,\
                                                    args.beam_width, args.eos)

            rec_pred = rec_pred.detach().cpu().numpy()
            test_pred.extend(rec_pred)
            test_image.extend(x.detach().cpu().numpy())

        test_pred_char = [self.idx2char(x, self.id2char_dict) for x in test_pred]

        return test_pred_char


if __name__ == "__main__":
    main_aster()
"""


if __name__ == "__main__":
    model = Pred_Aster()

    filenames = [x+'.xml' for x in file_val_list]

    from config import get_args
    args = get_args(sys.argv[1:])
    for i, filename in enumerate(filenames):
        if i < 10:
            image, coordinates, labels = get_data(filename, args)

            pred_char = model.forward(filename, coordinates)

            print(pred_char)
            print(labels)

"""



"""
ocr text in image
python ocr.py
"""
from PIL import Image
from PIL import ImageEnhance
#import pytesseract

import os
# tesseract executable file has to be installed beforehand
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#def get_ocr(image, coordinates,resample=Image.BICUBIC,resize=2,sharpness=1,contrast=1):

def get_ocr(image, coordinates, args, resample=Image.BICUBIC):
    """
    @Input
    image: an image (PIL)
    coordinates: a list of coordinates of text which should be recognized
                 It will be used for cropping specific area of image which contains text.
    
    @Output
    texts: a list of texts recognized from cropped image
    """
    
    cropped_images = crop_image(image,coordinates,resample, args)
    texts = []
    for i, cropped_image in enumerate(cropped_images):
        # if i == 0:
        #     # cropped_image.show()
        #for pagesegmode go to https://tesseract.patagames.com/help/html/T_Patagames_Ocr_Enums_PageSegMode.htm

        # Mod image into gray scale / binary scale
        if args.scale:
            cropped_image = cropped_image.convert(args.scale) # convert image to monochrome
        #cropped_image = cropped_image.convert('L') # convert image to monochrome
        #cropped_image = cropped_image.convert('1') # convert image to black/white

        text = pytesseract.image_to_string(cropped_image, config='--psm ' + args.psm) 
#        text = pytesseract.image_to_string(cropped_image, config='--psm ' + args.psm, lang='eng') 
#        text = pytesseract.image_to_string(cropped_image, config='--psm ' + args.psm, lang='wips') 
        #   tesseract with fine-tuned traineddata
#        tessdata_dir_config = r'--tessedata-dir "./data/fine-tuned"'
#        tessdata_dir_config = r'--tessedata-dir "data/tessdata_best-master" --psm '+args.psm
#        tessdata_dir_config = r'--tessedata-dir "/home/doyeong/wips/wips/src/data/tessdata_best-master" --psm '+args.psm
#        tessdata_dir_config = r'--tessedata-dir "/usr/share/tesseract-ocr/tessdata" --psm '+args.psm
#        text = pytesseract.image_to_string(cropped_image, lang='eng', config=tessdata_dir_config)
                                         
        texts.append(text)
        
    return texts

def crop_image(image,coordinates,resample, args):
    """
    @Input
    image: an image (PIL)
    coordinates: a list of coordinates of text which should be cropped
    
    @Output
    cropped_images: a list of cropped images
    """
    cropped_images = []
    for i,coordinate in enumerate(coordinates):
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

# OCR_Recognition

Required modules are:

```
python==3.6
torch==1.0.1
cv2==4.1.1
```

In order to check code running properly:

1. Download data.zip and params.zip from https://drive.google.com/drive/u/1/folders/1H3oiqq5g2kJutNuLZJvX4D58KnWGZI5a

2. Unzip files to the folder with same name respectively.

3. run run_model.sh to check model running.

In order to acquire fine-tuned parameters:

1. put datafolder into 'data' folder.

   formats of folders and their names follow 'data.zip'

   'Folder_name' of data folder is set to 'dataset_final' at the moment.

2. check 'Folder_name' in src/train_ocr_aster.py

3. run train_model.sh

   fine-tuned parameters of encoder/decoder are stored in 'params' folder.

   current name of parameter files are set to 'encoder_final'/'decoder_final'.



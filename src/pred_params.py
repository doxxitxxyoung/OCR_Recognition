class Get_ocr_args():
    def __init__(self):
#       self.cuda = False
        self.cuda = True # set to False when running on cpu
        self.eval = False # set to True when  model
        self.use_pretrained = True # set to False when you want models to be random-initialized


        self.seed = 44
        self.max_len = 18
        self.n_group = 1
        self.decoder_sdim = 512
        self.attDim = 512
        self.resize = 5
        self.sharpness = 1
        self.contrast = 3
        self.batch_size = 128
        self.height = 64
        self.width = 256
        self.beam_width = 5
        self.eos = 94
        self.image_format = 'pil'
        self.save_preds = False
        #   if training
        self.momentum = 0.9
        self.weight_decay = 0.0
        self.grad_clip = 1.0
        self.loss_weights = [1,1,1]
        self.n_epochs = 10
        self.batch_size = 128
        self.lr = 0.1
        self.rec_num_classes = 26
        self.STN_ON = False
        self.tps_inputsize = [32, 64]
        self.tps_outputsize = [32,100]
        self.num_control_points = 20
        self.tps_margins = [0.05, 0.05]
        self.stn_activation = 'none'




if __name__ == "__main__":
    args = Get_ocr_args()

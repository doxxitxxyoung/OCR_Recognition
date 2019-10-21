class Get_ocr_args():
    def __init__(self):
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

#       self.cuda = False
        self.cuda = True
        self.eval = False
        self.use_pretrained = False

        self.save_preds = False

        #   if training
        self.momentum = 0.9
        self.weight_decay = 0.0
        self.grad_clip = 1.0
        self.loss_weights = [1,1,1]
#        self.epochs = 6
        self.n_epochs = 10
        self.batch_size = 128
        self.lr = 0.1
        self.rec_num_classes = 26




if __name__ == "__main__":
    args = Get_ocr_args()

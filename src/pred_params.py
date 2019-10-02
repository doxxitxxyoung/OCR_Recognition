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
        self.cuda = False
        self.eval = False


if __name__ == "__main__":
    args = Get_ocr_args()

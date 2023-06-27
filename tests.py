import sys
import numpy as np
from model import RNN_ENCODER, G_NET, G_DCGAN
from model import D_NET64, D_NET128, D_NET256
from miscc.config import cfg
import torch
from PIL import Image
from torch.autograd import Variable
from torchsummary import summary
from torchvision import models
import torchvision.transforms as transforms
from BookCoverGenerator import getBookCovers, load_encoding, load_models, GeneratedImage
from NewTitleGenerator import generateNewTitles
from Vocabulary import loadVocab


if __name__ == '__main__':
    title = "adventure in a forest"

    _, wordtoix, n_words = load_encoding('captions.pickle')
    text_encoder, netG, netD = load_models(n_words)
    vocab = loadVocab()

    imgs = getBookCovers(title, wordtoix, netG, netD, text_encoder, vocab, 10)

    actualImg = imgs[0]
    imgs = imgs[1:]

    imgs.sort(key=lambda x: x.uncondScore, reverse=True)

    # imgs = imgs[0:5]

    imgs.insert(0, actualImg)

    for step, img in enumerate(imgs):
        img.img.save("output/img" + str(step) + ".png")
        print(img.title, img.condScore, img.uncondScore, str(step))



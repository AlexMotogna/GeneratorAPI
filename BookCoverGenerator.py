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
from NewTitleGenerator import generateNewTitles
from GeneratedImage import GeneratedImage


if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def load_encoding(pickle_file):
    x = pickle.load(open(pickle_file, 'rb'))
    ixtoword = x[2]
    wordtoix = x[3]
    n_words = len(ixtoword)
    del x

    return ixtoword, wordtoix, n_words


def load_models(n_words):
    if cfg.GAN.B_DCGAN:
        netD = D_NET256(b_jcu=False)
    else:
        netD = D_NET256()

    state_dict = torch.load(cfg.TRAIN.NET_D, map_location=lambda storage, loc: storage)
    netD.load_state_dict(state_dict)

    text_encoder = RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)

    if cfg.GAN.B_DCGAN:
        netG = G_DCGAN()
    else:
        netG = G_NET()
    state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
    netG.load_state_dict(state_dict)

    if cfg.CUDA:
        netG.cuda()
        text_encoder.cuda()
        netD.cuda()
    
    text_encoder.eval()
    netG.eval()
    netD.eval()

    return text_encoder, netG, netD


def vectorize_title(wordtoix, title):
    tokens = title.lower().split(' ')
    cap_v = []
    for t in tokens:
        t = t.encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0 and t in wordtoix:
            cap_v.append(wordtoix[t])

    captions = np.zeros((1, len(cap_v)))
    for i in range(1):
        captions[i,:] = np.array(cap_v)
    cap_lens = np.zeros(1) + len(cap_v)

    return captions.astype(int), cap_lens.astype(int)


def getDiscriminatorScore(netD, images, condition):
    features = netD(images.detach())
    cond_logits = netD.COND_DNET(features, condition)
    uncond_logits = netD.UNCOND_DNET(features)
    return cond_logits.item(), uncond_logits.item()


def generate(captions, cap_lenght, netG, netD, text_encoder):
    with torch.no_grad():
        batch_size = captions.shape[0]

        nz = cfg.GAN.Z_DIM
        captions = Variable(torch.from_numpy(captions))
        cap_lens = Variable(torch.from_numpy(cap_lenght))
        noise = Variable(torch.FloatTensor(batch_size, nz))

        if cfg.CUDA:
            captions = captions.cuda()
            cap_lens = cap_lens.cuda()
            noise = noise.cuda()

        hidden = text_encoder.init_hidden(batch_size)
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        mask = (captions == 0)

        noise.data.normal_(0, 1)
        fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)

        condScore, unCondScore = getDiscriminatorScore(netD, fake_imgs[2], sent_emb)

        im = fake_imgs[2][0].data.cpu().numpy()
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)

    return im, condScore, unCondScore


def getBookCoverByTitle(title, wordtoix, netG, netD, text_encoder):
    captions, cap_length = vectorize_title(wordtoix, title)
    im, condScore, unCondScore = generate(captions, cap_length, netG, netD, text_encoder)
    return GeneratedImage(im, title, condScore, unCondScore)


def getBookCovers(originalTitle, wordtoix, netG, netD, text_encoder, vocab, imgCount):
    newTitles = generateNewTitles(originalTitle, vocab, imgCount)
    
    imgs = []

    for title in newTitles:
        result = getBookCoverByTitle(title, wordtoix, netG, netD, text_encoder)
        imgs.append(result)

    return imgs

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
    features = netD(images)
    cond_logits = netD.COND_DNET(features, condition)
    uncond_logits = netD.UNCOND_DNET(features)
    
    print(cond_logits)
    print(uncond_logits)


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
        # print(noise.size())
        # print(sent_emb.size())
        # print(words_embs.size())
        # print(mask.size())
        fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)

        getDiscriminatorScore(netD, fake_imgs[2].detach(), sent_emb)


        for j in range(batch_size):
            for k in range(len(fake_imgs)):
                im = fake_imgs[k][j]
                mean = 0.
                std = 0.05
                gaussnoise = (torch.randn(im.size())) * std
                im = im + gaussnoise

                im = im.data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                fullpath = '%s_g%d.png' % ("output/img", k)
                im.save(fullpath)

    return


def getRealPictures():
    img = Image.open("data/0-252577.jpg").convert('RGB')
    
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    
    img = image_transform(img)

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    sizes = []
    base_size=cfg.TREE.BASE_SIZE
    for i in range(cfg.TREE.BRANCH_NUM):
        sizes.append(base_size)
        base_size = base_size * 2
    
    aux = []
    if cfg.GAN.B_DCGAN:
        aux = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Resize(sizes[i])(img)
            else:
                re_img = img
            aux.append(normalize(re_img))

    real_imgs = []
    for i in range(len(aux)):
        # aux[i] = aux[i][0]
        if cfg.CUDA:
            real_imgs.append(Variable(aux[i]).to(0))
        else:
            real_imgs.append(Variable(aux[i]))

    return aux


def evaluateRealPicture(captions, cap_length, netD, imgs):
    captions = Variable(torch.from_numpy(captions))
    cap_lens = Variable(torch.from_numpy(cap_length))

    hidden = text_encoder.init_hidden(captions.shape[0])
    _, sent_emb = text_encoder(captions, cap_lens, hidden)

    imgs = getRealPictures()
    getDiscriminatorScore(netD, imgs[2], sent_emb)
    return


if __name__ == '__main__':
    title = "dragon fire"

    ixtoword, wordtoix, n_words = load_encoding('captions.pickle')
    text_encoder, netG, netD = load_models(n_words)

    captions, cap_length = vectorize_title(wordtoix, title)
    generate(captions, cap_length, netG, netD, text_encoder)



import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from miscc.config import cfg, cfg_from_file
from miscc.utils import mkdir_p
from miscc.losses import discriminator_loss, mask_loss, KL_loss,TV_loss
from datasets import TextDataset
from datasets import prepare_data
from model import G_MASK, RNN_ENCODER
from zzbb import D_NET64
from tensorboardX import SummaryWriter


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_mask.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

def build_models(n_words):
    # build model ############################################################
    text_encoder = \
        RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = \
        torch.load(cfg.TRAIN.NET_E,
                   map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    for p in text_encoder.parameters():
        p.requires_grad = False
    print('Load text encoder from:', cfg.TRAIN.NET_E)
    text_encoder.eval()
    G = G_MASK()
    D = D_NET64()
    start_epoch = 0
    if cfg.CUDA:
        text_encoder = text_encoder.cuda()
        G = G.cuda()
        D = D.cuda()

    return text_encoder, G, D, start_epoch

def define_optimizers(netG, netD):
    optimizerG = optim.Adam(netG.parameters(),
                            lr=cfg.TRAIN.DISCRIMINATOR_LR,
                            betas=(0.5, 0.999))

    optimizerD = optim.Adam(netD.parameters(),
                            lr=cfg.TRAIN.GENERATOR_LR,
                            betas=(0.5, 0.999))
    return optimizerG, optimizerD

def prepare_labels(batch_size):
    real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
    fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
    match_labels = Variable(torch.LongTensor(range(batch_size)))
    if cfg.CUDA:
        real_labels = real_labels.cuda()
        fake_labels = fake_labels.cuda()
        match_labels = match_labels.cuda()

    return real_labels, fake_labels, match_labels


def save_model(net, opt, epoch, name, model_dir="models"):
    statu = {
        'epoch': epoch,
        'model': net.state_dict(),
        'optimizer': opt.state_dict()
    }
    torch.save(statu, '%s/net%s_epoch_%d.pth' % (model_dir, name, epoch))

def load_model(net, opt, name):
    state_dict = \
        torch.load(name, map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict['model'])
    opt.load_state_dict(state_dict['optimizer'])
    istart = name.rfind('_') + 1
    iend = name.rfind('.')
    epoch = name[istart:iend]
    epoch = int(epoch) + 1
    return net, opt, epoch

def train(output_dir, data_loader, n_words):
    model_dir = os.path.join(output_dir, 'Model')
    log_dir = os.path.join(output_dir, 'logs')
    mkdir_p(model_dir)
    mkdir_p(log_dir)
    writer = SummaryWriter(log_dir)
    batch_size = cfg.TRAIN.BATCH_SIZE

    text_encoder, netG, netD, start_epoch = build_models(n_words)
    opt_G, opt_D = define_optimizers(netG, netD)
    real_labels, fake_labels, match_labels = prepare_labels(batch_size)
    if cfg.TRAIN.NET_G != '':
        netG, opt_G, start_epoch = load_model(netG, opt_G, cfg.TRAIN.NET_G)
        netD, opt_D, _ = load_model(netD, opt_D, cfg.TRAIN.NET_G.replace("netG, netD"))

    nz = cfg.GAN.Z_DIM
    noise = Variable(torch.FloatTensor(batch_size, nz))
    fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
    if cfg.CUDA:
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    gen_iterations = 0
    # gen_iterations = start_epoch * self.num_batches
    for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
        start_t = time.time()

        data_iter = iter(data_loader)
        step = 0
        while step < len(data_loader):
            # reset requires_grad to be trainable for all Ds
            # self.set_requires_grad_value(netsD, True)

            ######################################################
            # (1) Prepare training data and Compute text embeddings
            ######################################################
            data = data_iter.next()
            imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

            hidden = text_encoder.init_hidden(batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            # words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

            #######################################################
            # (2) Generate fake images
            ######################################################
            noise.data.normal_(0, 1)
            fake_imgs, mu, logvar = netG(noise, sent_emb)

            #######################################################
            # (3) Update D network
            ######################################################
            errD_total = 0
            netD.zero_grad()
            # print(imgs[i].shape, fake_imgs[i].shape, sent_emb.shape, real_labels.shape, fake_labels.shape)
            errD, pred_loss = discriminator_loss(netD, imgs[0], fake_imgs, sent_emb, real_labels, fake_labels)
            # backward and update parameters
            errD.backward()
            opt_D.step()
            D_logs = 'errD: %.2f ' % (errD.item())

            #######################################################
            # (4) Update G network: maximize log(D(G(z)))
            ######################################################
            # compute total loss for training G
            step += 1
            gen_iterations += 1

            # do not need to compute gradient for Ds
            # self.set_requires_grad_value(netsD, False)
            # if gen_iterations % 3 == 0:
            netG.zero_grad()
            text_encoder.zero_grad()
            errG_total, G_logs, errG_list = mask_loss(netD, fake_imgs, real_labels,
                               words_embs, sent_emb, match_labels, cap_lens, class_ids)
            kl_loss = KL_loss(mu, logvar)
            # tv_loss = TV_loss(fake_imgs)
            errG_total += kl_loss
            # errG_total += tv_loss#add tv loss
            G_logs += 'kl_loss: %.2f ' % kl_loss.item()
            # G_logs += 'tv_loss: %.2f ' % tv_loss.data[0]
            # backward and update parameters
            errG_total.backward()
            opt_G.step()

            if gen_iterations % 10 == 0:
                writer.add_scalar("watch/errD", errD, gen_iterations)
                writer.add_scalar("watch/errG", errG_total, gen_iterations)
                writer.add_scalar("watch/real", pred_loss[0], gen_iterations)
                writer.add_scalar("watch/fake", pred_loss[1], gen_iterations)
                writer.add_image('real_image', imgs[0][0], gen_iterations)
                writer.add_image('fake_image', fake_imgs[0], gen_iterations)

            if gen_iterations % 100 == 0:
                print(D_logs + '\n' + G_logs)

        end_t = time.time()

        print('''[%d/%d][%d] Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
              % (epoch, cfg.TRAIN.MAX_EPOCH, len(data_loader), errD.item(), errG_total.item(), end_t - start_t))

        if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
            save_model(netG, opt_G, epoch, "G", model_dir)
            save_model(netD, opt_D, epoch, "D", model_dir)

    save_model(netG, opt_G, epoch, "G", model_dir)
    save_model(netD, opt_D, epoch, "D", model_dir)


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/MASK_%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bshuffle = False
        split_dir = 'test'

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform,
                          name='segmentations')
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))


    train(output_dir, dataloader, dataset.n_words)

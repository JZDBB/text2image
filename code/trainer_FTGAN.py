from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2, image_comp
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET
from datasets import prepare_data
from model import RNN_ENCODER, CNN_ENCODER

from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss,TV_loss, mask_loss
import os
import time
import numpy as np
import sys
from tensorboardX import SummaryWriter

# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword, args):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, "logs")
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)

        if cfg.CUDA:
            torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
        self.args = args

    def build_models(self):
        # ###################encoders######################################## #
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        text_encoder = \
            RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E,
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        # #######################generator and discriminators############## #
        netsD = []
        if cfg.GAN.B_DCGAN:
            if cfg.TREE.BRANCH_NUM ==1:
                from model import D_NET64 as D_NET
            elif cfg.TREE.BRANCH_NUM == 2:
                from model import D_NET128 as D_NET
            else:  # cfg.TREE.BRANCH_NUM == 3:
                from model import D_NET256 as D_NET
            # TODO: elif cfg.TREE.BRANCH_NUM > 3:
            netG = G_DCGAN()
            netsD = [D_NET(b_jcu=False)]
        else:
            from model import D_NET64, D_NET128, D_NET256, D_MASK
            netG = G_NET()
            if cfg.TREE.BRANCH_NUM > 0:
                netsD.append(D_NET64())
            if cfg.TREE.BRANCH_NUM > 1:
                netsD.append(D_NET128())
            if cfg.TREE.BRANCH_NUM > 2:
                netsD.append(D_NET256())
            # TODO: if cfg.TREE.BRANCH_NUM > 3:
            maskD = D_MASK()
        netG.apply(weights_init)
        # print(netG)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
            # print(netsD[i])
        print('# of netsD', len(netsD))
        #
        epoch = 0

        return [text_encoder, image_encoder, netG, netsD, maskD, epoch]

    def define_optimizers(self, netG, netsD, mask_D):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        # optimizerE = optim.Adam(text_encoder.parameters(),
        #                         lr=cfg.TRAIN.GENERATOR_LR*0.5,
        #                         betas=(0.5, 0.999))

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        optimizerM = optim.Adam(mask_D.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        return optimizerG, optimizersD, optimizerM

    def adjust_learning_rate(self, optimizer, decay_rate=.9):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate

    def load_model(self, netG, netsD, mask_D):
        state_dict = \
            torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
        netG.load_state_dict(state_dict['model'])
        self.optimizerG.load_state_dict(state_dict['optimizer'])
        print('Load G from: ', cfg.TRAIN.NET_G)

        istart = cfg.TRAIN.NET_G.rfind('_') + 1
        iend = cfg.TRAIN.NET_G.rfind('.')
        epoch = cfg.TRAIN.NET_G[istart:iend]
        epoch = int(epoch) + 1
        Gname = cfg.TRAIN.NET_G
        s_tmp = Gname[:Gname.rfind('/')]
        maskDname = '%s/netD_mask.pth' % (s_tmp)
        state_dict = \
            torch.load(maskDname, map_location=lambda storage, loc: storage)
        mask_D.load_state_dict(state_dict['model'])
        self.optimizerM.load_state_dict(state_dict['optimizer'])
        print('Load D_MASK from: ', maskDname)

        if cfg.TRAIN.B_NET_D:
            for i in range(len(netsD)):
                s_tmp = Gname[:Gname.rfind('/')]
                Dname = '%s/netD%d.pth' % (s_tmp, i)
                print('Load D from: ', Dname)
                state_dict = \
                    torch.load(Dname, map_location=lambda storage, loc: storage)
                netsD[i].load_state_dict(state_dict['model'])
                self.optimizersD[i].load_state_dict(state_dict['optimizer'])

        return netG, netsD, mask_D, epoch

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def compose_image(self, mask_r, mask_f, real_images, fake_images):
        reals = []
        fakes = []
        size = [64, 128, 256]
        real_mask = image_comp(mask_r[0], 64, self.batch_size, 1)
        fake_mask = image_comp(mask_f, 64, self.batch_size, 1)
        for i in range(len(real_images)):
            real = image_comp(real_images[i], size[i], self.batch_size, 3)
            reals.append(real)
            fake = image_comp(fake_images[i], size[i], self.batch_size, 3)
            fakes.append(fake)

        return real_mask, fake_mask, reals, fakes

    def save_model(self, netG, avg_param_G, netsD, mask_D, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        statu = {
            'epoch': epoch,
            'model': netG.state_dict(),
            'optimizer': self.optimizerG.state_dict()
        }
        torch.save(statu, '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        statu = {
            'epoch': epoch,
            'model': mask_D.state_dict(),
            'optimizer': self.optimizerM.state_dict()
        }
        torch.save(statu, '%s/netD_mask.pth' % (self.model_dir))
        for i in range(len(netsD)):
            statu = {
                'epoch': epoch,
                'model': netsD[i].state_dict(),
                'optimizer': self.optimizersD[i].state_dict()
            }
            torch.save(statu, '%s/netD%d.pth' % (self.model_dir, i))
        print('Save G/Ds models.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, name='current'):
        # Save images
        fake_imgs, _, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png'\
                    % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)
        # self.writer.add_images("low res image maps", img_set.datach())

        # for i in range(len(netsD)):
        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, self.ixtoword, att_maps, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)

        # self.writer.add_images("high res image maps", im)


    def train(self):
        self.writer = SummaryWriter(self.log_dir)
        text_encoder, image_encoder, netG, netsD, mask_D, start_epoch = self.build_models()

        if cfg.TRAIN.NET_G != '':
            netG, netsD, mask_D, start_epoch = self.load_model(netG, netsD, mask_D)
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            netG.cuda()
            mask_D.cuda()
            for i in range(len(netsD)):
                netsD[i].cuda()
        self.optimizerG, self.optimizersD, self.optimizerM = self.define_optimizers(netG, netsD, mask_D)
        avg_param_G = copy_G_params(netG)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, masks, captions, cap_lens, class_ids, keys = prepare_data(data)

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                #words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, mask_imgs, _, mu, logvar = netG(noise, sent_emb, words_embs, mask)

                #######################################################
                # (3) Update D network
                ######################################################

                errD_total = 0
                D_logs = ''
                errD_ = []
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                              sent_emb, real_labels, fake_labels)
                    # backward and update parameters
                    errD.backward()
                    self.optimizersD[i].step()
                    errD_total += errD
                    errD_.append(errD)
                    D_logs += 'errD%d: %.2f ' % (i, errD.item())

                mask_D.zero_grad()
                err_mask = mask_loss(mask_D, masks[0], mask_imgs,
                                              sent_emb, real_labels, fake_labels)
                err_mask.backward()
                self.optimizerM.step()
                errD_total += err_mask
                D_logs += 'maskD: %.2f ' % (err_mask.item())

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
                # text_encoder.zero_grad()
                errG_total, G_logs, errG_list = \
                    generator_loss(netsD, mask_D, image_encoder, fake_imgs, mask_imgs, real_labels,
                                   words_embs, sent_emb, match_labels, cap_lens, class_ids)
                kl_loss = KL_loss(mu, logvar)
                #tv_loss = TV_loss(fake_imgs)
                errG_total += kl_loss
                #errG_total += tv_loss#add tv loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.item()
                # backward and update parameters
                errG_total.backward()
                self.optimizerG.step()
				
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 50 == 0:
                    self.writer.add_scalar("watch/errD0", errD_[0], gen_iterations)
                    self.writer.add_scalar("watch/errD1", errD_[1], gen_iterations)
                    self.writer.add_scalar("watch/errD2", errD_[2], gen_iterations)
                    self.writer.add_scalar("watch/mask_D", err_mask, gen_iterations)
                    self.writer.add_scalar("watch/errD_total", errD_total, gen_iterations)
                    self.writer.add_scalar("watch/errg0", errG_list[0], gen_iterations)
                    self.writer.add_scalar("watch/errg1", errG_list[1], gen_iterations)
                    self.writer.add_scalar("watch/errg2", errG_list[2], gen_iterations)
                    self.writer.add_scalar("watch/word_loss", errG_list[3], gen_iterations)
                    self.writer.add_scalar("watch/sent_loss", errG_list[4], gen_iterations)
                    self.writer.add_scalar("watch/mask_loss", errG_list[5], gen_iterations)
                    self.writer.add_scalar("watch/errG", errG_total, gen_iterations)
                    # self.writer.add_scalar("watch/p_real", errG_total, gen_iterations)
                    # self.writer.add_scalar("watch/p_fake", errG_total, gen_iterations)
                    # self.writer.add_scalar("watch/learning_rate", optimizerG['lr'], gen_iterations)
                    mask_r, mask_f, reals, fakes = self.compose_image(masks, mask_imgs, imgs, fake_imgs)
                    self.writer.add_image('mask_r', mask_r, gen_iterations)
                    self.writer.add_image('mask_f', mask_f, gen_iterations)
                    self.writer.add_image('fake1', fakes[0], gen_iterations)
                    self.writer.add_image('fake2', fakes[1], gen_iterations)
                    self.writer.add_image('fake3', fakes[2], gen_iterations)
                    self.writer.add_image('real1', reals[0], gen_iterations)
                    self.writer.add_image('real2', reals[1], gen_iterations)
                    self.writer.add_image('real3', reals[2], gen_iterations)

                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + G_logs)
                # save images
                # if gen_iterations % 1000 == 0:
                #     backup_para = copy_G_params(netG)
                #     load_params(netG, avg_param_G)
                #     self.save_img_results(netG, fixed_noise, sent_emb,
                #                           words_embs, mask, image_encoder,
                #                           captions, cap_lens, epoch, name='average')
                #     load_params(netG, backup_para)
                    #
                    # self.save_img_results(netG, fixed_noise, sent_emb,
                    #                       words_embs, mask, image_encoder,
                    #                       captions, cap_lens,
                    #                       epoch, name='current')
                # if optimizerG['lr'] > 2e-7:
                #     if epoch < 100:
                #         self.adjust_learning_rate(optimizerE, 1.1)
                #         self.adjust_learning_rate(optimizerG, 1.1)
                #         for opt in optimizersD:
                #             self.adjust_learning_rate(opt, 1.1)
                #     else:
                #         self.adjust_learning_rate(optimizerE, 0.99)
                #         self.adjust_learning_rate(optimizerG, 0.99)
                #     for opt in optimizersD:
                #         self.adjust_learning_rate(opt, 0.99)

            end_t = time.time()

            print('''[%d/%d][%d] Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch,  self.max_epoch, self.num_batches, errD_total.item(), errG_total.item(), end_t - start_t))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsD, mask_D, epoch)

        self.save_model(netG, avg_param_G, netsD, mask_D, self.max_epoch)
        self.writer.close()

    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def sampling(self, split_dir):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            netG.cuda()
            netG.eval()
            #
            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise = noise.cuda()

            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s/%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0
            for _ in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                for step, data in enumerate(self.data_loader, 0):
                    cnt += batch_size
                    if step % 100 == 0:
                        print('step: ', step)
                    # if step > 50:
                    #     break

                    imgs, _, captions, cap_lens, class_ids, keys = prepare_data(data)

                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, _, _, _, _ = netG(noise, sent_emb, words_embs, mask)
                    for j in range(batch_size):
                        s_tmp = '%s/single/%s' % (save_dir, keys[j])
                        folder = s_tmp[:s_tmp.rfind('/')]
                        if not os.path.isdir(folder):
                            print('Make a new folder: ', folder)
                            mkdir_p(folder)
                        k = -1
                        # for k in range(len(fake_imgs)):
                        im = fake_imgs[k][j].data.cpu().numpy()
                        # [-1, 1] --> [0, 255]
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        fullpath = '%s_s%d.png' % (s_tmp, k)
                        im.save(fullpath)

    def sampling_n(self, split_dir):
        if cfg.TRAIN.NET_G == '':
            if self.args.model is None:
                print('Error: the path for morels is not found!')
                return
            else:
                model_dir = self.args.model
        else:
            model_dir = cfg.TRAIN.NET_G

        if split_dir == 'test':
            split_dir = 'valid'
        # Build and load the generator
        if cfg.GAN.B_DCGAN:
            netG = G_DCGAN()
        else:
            netG = G_NET()
        netG.apply(weights_init)
        netG.cuda()
        netG.eval()
        #
        text_encoder_dir = cfg.TRAIN.NET_E
        text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(text_encoder_dir, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        print('Load text encoder from:', text_encoder_dir)
        text_encoder = text_encoder.cuda()
        text_encoder.eval()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
        noise = noise.cuda()

        state_dict = \
            torch.load(model_dir, map_location=lambda storage, loc: storage)
        # state_dict = torch.load(cfg.TRAIN.NET_G)
        netG.load_state_dict(state_dict['model'])
        print('Load G from: ', model_dir)

        # the path to save generated images
        s_tmp = model_dir[:model_dir.rfind('.pth')]
        save_dir = '%s/%s' % (s_tmp, split_dir)
        mkdir_p(save_dir)

        cnt = 0
        for cn in range(10):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
            for step, data in enumerate(self.data_loader, 0):
                cnt += batch_size
                if step % 100 == 0:
                    print('step: ', step)
                # if step > 50:
                #     break

                imgs, _, captions, cap_lens, class_ids, keys = prepare_data(data)

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, _, _, _, _ = netG(noise, sent_emb, words_embs, mask)
                for j in range(batch_size):
                    s_tmp = '%s/multiple/%s' % (save_dir, keys[j])
                    folder = s_tmp[:s_tmp.rfind('/')]
                    if not os.path.isdir(folder):
                        print('Make a new folder: ', folder)
                        mkdir_p(folder)
                    k = -1 #-1
                    # for k in range(len(fake_imgs)):
                    im = fake_imgs[k][j].data.cpu().numpy()
                    # [-1, 1] --> [0, 255]
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))
                    im = Image.fromarray(im)
                    fullpath = '%s_s%d.png' % (s_tmp, k-cn)
                    im.save(fullpath)

    def gen_example(self, data_dic):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            text_encoder = \
                RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            # the path to save generated images
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            netG.cuda()
            netG.eval()
            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices = data_dic[key]

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM
                captions = Variable(torch.from_numpy(captions))#, volatile=True)
                cap_lens = Variable(torch.from_numpy(cap_lens))#, volatile=True)

                captions = captions.cuda()
                cap_lens = cap_lens.cuda()
                for i in range(1):  # 16
                    noise = Variable(torch.FloatTensor(batch_size, nz))#, volatile=True)
                    noise = noise.cuda()
                    #######################################################
                    # (1) Extract text embeddings
                    ######################################################
                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    mask = (captions == 0)
                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    fake_imgs, _, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                    # G attention
                    cap_lens_np = cap_lens.cpu().data.numpy()
                    for j in range(batch_size):
                        save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            # print('im', im.shape)
                            im = np.transpose(im, (1, 2, 0))
                            # print('im', im.shape)
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)

                        for k in range(len(attention_maps)):
                            if len(fake_imgs) > 1:
                                im = fake_imgs[k + 1].detach().cpu()
                            else:
                                im = fake_imgs[0].detach().cpu()
                            attn_maps = attention_maps[k]
                            att_sze = attn_maps.size(2)
                            img_set, sentences = \
                                build_super_images2(im[j].unsqueeze(0),
                                                    captions[j].unsqueeze(0),
                                                    [cap_lens_np[j]], self.ixtoword,
                                                    [attn_maps[j]], att_sze)
                            if img_set is not None:
                                im = Image.fromarray(img_set)
                                fullpath = '%s_a%d.png' % (save_name, k)
                                im.save(fullpath)

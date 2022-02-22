"""Main"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib
#matplotlib.use('agg')
from matplotlib import pyplot as plt
plt.ion()
import os

from dataset.dataset_class import PreprocessDataset
from dataset.video_extraction_conversion import *
from loss.loss_discriminator import *
from loss.loss_generator import *
from network.blocks import *
from network.model import *
from tqdm import tqdm

from params.params import K, path_to_chkpt, path_to_backup, path_to_Wi, batch_size, path_to_preprocess, frame_shape, VGG19_caffe_weight_path

"""Create dataset and net"""
display_training = False
device = torch.device("cuda:0")
cpu = torch.device("cpu")
dataset = PreprocessDataset(K=K, path_to_preprocess=path_to_preprocess, path_to_Wi=path_to_Wi)
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=16,
                        pin_memory=True,
                        drop_last = True)

G = nn.DataParallel(Generator(frame_shape).to(device))
E = nn.DataParallel(Embedder(frame_shape).to(device))
D = nn.DataParallel(Discriminator(dataset.__len__(), path_to_Wi).to(device))

G.train()
E.train()
D.train()


optimizerG = optim.Adam(params = list(E.parameters()) + list(G.parameters()),
                        lr=5e-5,
                        amsgrad=False)
optimizerD = optim.Adam(params = D.parameters(),
                        lr=2e-4,
                        amsgrad=False)

"""Criterion"""
criterionG = LossG(VGGFace_body_path='Pytorch_VGGFACE_IR.py',
                   VGGFace_weight_path='Pytorch_VGGFACE.pth',
                   VGG19_CAFFE_weight_path=VGG19_caffe_weight_path,
                device=device)
criterionDreal = LossDSCreal()
criterionDfake = LossDSCfake()


"""Training init"""
epochCurrent = epoch = i_batch = 0
lossesG = []
lossesD = []
i_batch_current = 0

num_epochs = 75*5

#initiate checkpoint if inexistant
if not os.path.isfile(path_to_chkpt):
    def init_weights(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform(m.weight)
    G.apply(init_weights)
    D.apply(init_weights)
    E.apply(init_weights)

    print('Initiating new checkpoint...')
    torch.save({
            'epoch': epoch,
            'lossesG': lossesG,
            'lossesD': lossesD,
            'E_state_dict': E.module.state_dict(),
            'G_state_dict': G.module.state_dict(),
            'D_state_dict': D.module.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': i_batch,
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict()
            }, path_to_chkpt)
    print('...Done')


"""Loading from past checkpoint"""
checkpoint = torch.load(path_to_chkpt, map_location=cpu)
E.module.load_state_dict(checkpoint['E_state_dict'])
G.module.load_state_dict(checkpoint['G_state_dict'], strict=False)
D.module.load_state_dict(checkpoint['D_state_dict'])
epochCurrent = checkpoint['epoch']
lossesG = checkpoint['lossesG']
lossesD = checkpoint['lossesD']
num_vid = checkpoint['num_vid']
i_batch_current = checkpoint['i_batch'] +1
optimizerG.load_state_dict(checkpoint['optimizerG'])
optimizerD.load_state_dict(checkpoint['optimizerD'])

G.train()
E.train()
D.train()

"""Training"""
batch_start = datetime.now()
pbar = tqdm(dataLoader, leave=True, initial=0)
if not display_training:
    matplotlib.use('agg')


for epoch in range(epochCurrent, num_epochs):
    if epoch > epochCurrent:
        i_batch_current = 0
        pbar = tqdm(dataLoader, leave=True, initial=0)
    pbar.set_postfix(epoch=epoch)
    for i_batch, (f_lm, x, g_y, i, W_i) in enumerate(pbar, start=0):
        
        f_lm = f_lm.to(device)
        x = x.to(device)
        g_y = g_y.to(device)
        W_i = W_i.squeeze(-1).transpose(0,1).to(device).requires_grad_()
        
        D.module.load_W_i(W_i)
        
        if i_batch % 1 == 0:
            with torch.autograd.enable_grad():
                #zero the parameter gradients
                optimizerG.zero_grad()
                optimizerD.zero_grad()

                #forward
                # Calculate average encoding vector for video
                f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]) #BxK,2,3,224,224

                e_vectors = E(f_lm_compact[:,0,:,:,:], f_lm_compact[:,1,:,:,:]) #BxK,512,1
                e_vectors = e_vectors.view(-1, f_lm.shape[1], 512, 1) #B,K,512,1
                e_hat = e_vectors.mean(dim=1)

                #train G and D
                x_hat = G(g_y, e_hat)
                r_hat, D_hat_res_list = D(x_hat, g_y, i)
                with torch.no_grad():
                    r, D_res_list = D(x, g_y, i)
                """####################################################################################################################################################
                r, D_res_list = D(x, g_y, i)"""

                lossG = criterionG(x, x_hat, r_hat, D_res_list, D_hat_res_list, e_vectors, D.module.W_i, i)
                
                """####################################################################################################################################################
                lossD = criterionDfake(r_hat) + criterionDreal(r)
                loss = lossG + lossD
                loss.backward(retain_graph=False)
                optimizerG.step()
                optimizerD.step()"""
                
                lossG.backward(retain_graph=False)
                optimizerG.step()
                #optimizerD.step()
            
            with torch.autograd.enable_grad():
                optimizerG.zero_grad()
                optimizerD.zero_grad()
                x_hat.detach_().requires_grad_()
                r_hat, D_hat_res_list = D(x_hat, g_y, i)
                lossDfake = criterionDfake(r_hat)

                r, D_res_list = D(x, g_y, i)
                lossDreal = criterionDreal(r)
                
                lossD = lossDfake + lossDreal
                lossD.backward(retain_graph=False)
                optimizerD.step()
                #for p in D.module.parameters():
                 #   p.data.clamp_(-1.0, 1.0)
                
                
                optimizerD.zero_grad()
                r_hat, D_hat_res_list = D(x_hat, g_y, i)
                lossDfake = criterionDfake(r_hat)

                r, D_res_list = D(x, g_y, i)
                lossDreal = criterionDreal(r)
                
                lossD = lossDfake + lossDreal
                lossD.backward(retain_graph=False)
                optimizerD.step()
                #for p in D.module.parameters():
                 #   p.data.clamp_(-1.0, 1.0)

        for enum, idx in enumerate(i):
            torch.save({'W_i': D.module.W_i[:,enum].unsqueeze(-1)}, path_to_Wi+'/W_'+str(idx.item())+'/W_'+str(idx.item())+'.tar')
                    

        # Output training stats
        if i_batch % 1 == 0 and i_batch > 0:
            #batch_end = datetime.now()
            #avg_time = (batch_end - batch_start) / 100
            # print('\n\navg batch time for batch size of', x.shape[0],':',avg_time)
            
            #batch_start = datetime.now()
            
            # print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(y)): %.4f'
            #       % (epoch, num_epochs, i_batch, len(dataLoader),
            #          lossD.item(), lossG.item(), r.mean(), r_hat.mean()))
            pbar.set_postfix(epoch=epoch, r=r.mean().item(), rhat=r_hat.mean().item(), lossG=lossG.item())

            if display_training:
                plt.figure(figsize=(10,10))
                plt.clf()
                out = (x_hat[0]*255).transpose(0,2)
                for img_no in range(1,x_hat.shape[0]//16):
                    out = torch.cat((out, (x_hat[img_no]*255).transpose(0,2)), dim = 1)
                out = out.type(torch.int32).to(cpu).numpy()
                fig = out

                plt.clf()
                out = (x[0]*255).transpose(0,2)
                for img_no in range(1,x.shape[0]//16):
                    out = torch.cat((out, (x[img_no]*255).transpose(0,2)), dim = 1)
                out = out.type(torch.int32).to(cpu).numpy()
                fig = np.concatenate((fig, out), 0)

                plt.clf()
                out = (g_y[0]*255).transpose(0,2)
                for img_no in range(1,g_y.shape[0]//16):
                    out = torch.cat((out, (g_y[img_no]*255).transpose(0,2)), dim = 1)
                out = out.type(torch.int32).to(cpu).numpy()
                
                fig = np.concatenate((fig, out), 0)
                plt.imshow(fig)
                plt.xticks([])
                plt.yticks([])
                plt.draw()
                plt.pause(0.001)
            
            

        if i_batch % 1000 == 999:
            lossesD.append(lossD.item())
            lossesG.append(lossG.item())

            if display_training:
                plt.clf()
                plt.plot(lossesG) #blue
                plt.plot(lossesD) #orange
                plt.show()

            print('Saving latest...')
            torch.save({
                    'epoch': epoch,
                    'lossesG': lossesG,
                    'lossesD': lossesD,
                    'E_state_dict': E.module.state_dict(),
                    'G_state_dict': G.module.state_dict(),
                    'D_state_dict': D.module.state_dict(),
                    'num_vid': dataset.__len__(),
                    'i_batch': i_batch,
                    'optimizerG': optimizerG.state_dict(),
                    'optimizerD': optimizerD.state_dict()
                    }, path_to_chkpt)
            out = (x_hat[0]*255).transpose(0,2)
            for img_no in range(1,2):
                out = torch.cat((out, (x_hat[img_no]*255).transpose(0,2)), dim = 1)
            out = out.type(torch.uint8).to(cpu).numpy()
            plt.imsave("recent.png", out)
            print('...Done saving latest')
            
    if epoch%1 == 0:
        print('Saving latest...')
        torch.save({
                'epoch': epoch+1,
                'lossesG': lossesG,
                'lossesD': lossesD,
                'E_state_dict': E.module.state_dict(),
                'G_state_dict': G.module.state_dict(),
                'D_state_dict': D.module.state_dict(),
                'num_vid': dataset.__len__(),
                'i_batch': i_batch,
                'optimizerG': optimizerG.state_dict(),
                'optimizerD': optimizerD.state_dict()
                }, path_to_backup)
        out = (x_hat[0]*255).transpose(0,2)
        for img_no in range(1,2):
            out = torch.cat((out, (x_hat[img_no]*255).transpose(0,2)), dim = 1)
        out = out.type(torch.uint8).to(cpu).numpy()
        plt.imsave("recent_backup.png", out)
        print('...Done saving latest')

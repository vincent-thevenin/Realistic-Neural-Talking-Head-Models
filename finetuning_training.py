import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib

import numpy as np

from dataset.dataset_class import FineTuningImagesDataset, FineTuningVideoDataset
from network.model import *
from loss.loss_discriminator import *
from loss.loss_generator import *

from params.params import K, path_to_chkpt, path_to_backup, path_to_Wi, batch_size, path_to_preprocess, frame_shape

"""Hyperparameters and config"""
display_training = True
if not display_training:
	matplotlib.use('agg')
device = torch.device("cuda:0")
cpu = torch.device("cpu")
path_to_embedding = 'e_hat_video.tar'
path_to_save = 'finetuned_model.tar'
path_to_video = 'examples/fine_tuning/test_video.mp4'
path_to_images = 'examples/fine_tuning/test_images'


"""Create dataset and net"""
choice = ''
while choice != '0' and choice != '1':
    choice = input('What source to finetune on?\n0: Video\n1: Images\n\nEnter number\n>>')
if choice == '0': #video
    dataset = FineTuningVideoDataset(path_to_video, device)
else: #Images
    dataset = FineTuningImagesDataset(path_to_images, device)
dataLoader = DataLoader(dataset, batch_size=2, shuffle=False)

e_hat = torch.load(path_to_embedding, map_location=cpu)
e_hat = e_hat['e_hat']

G = Generator(256, finetuning = True, e_finetuning = e_hat)
D = Discriminator(dataset.__len__(), path_to_Wi, finetuning = True, e_finetuning = e_hat)

G.train()
D.train()

optimizerG = optim.Adam(params = G.parameters(), lr=5e-5)
optimizerD = optim.Adam(params = D.parameters(), lr=2e-4)


"""Criterion"""
criterionG = LossGF(VGGFace_body_path='Pytorch_VGGFACE_IR.py',
                   VGGFace_weight_path='Pytorch_VGGFACE.pth', device=device)
criterionDreal = LossDSCreal()
criterionDfake = LossDSCfake()


"""Training init"""
epochCurrent = epoch = i_batch = 0
lossesG = []
lossesD = []
i_batch_current = 0

num_epochs = 40

#Warning if checkpoint inexistant
if not os.path.isfile(path_to_chkpt):
    print('ERROR: cannot find checkpoint')
if os.path.isfile(path_to_save):
    path_to_chkpt = path_to_save

"""Loading from past checkpoint"""
checkpoint = torch.load(path_to_chkpt, map_location=cpu)
checkpoint['D_state_dict']['W_i'] = torch.rand(512, 32) #change W_i for finetuning

G.load_state_dict(checkpoint['G_state_dict'])
D.load_state_dict(checkpoint['D_state_dict'], strict = False)


"""Change to finetuning mode"""
G.finetuning_init()
D.finetuning_init()

G.to(device)
D.to(device)

"""Training"""
batch_start = datetime.now()

cont = True
while cont:
    for epoch in range(num_epochs):
        for i_batch, (x, g_y) in enumerate(dataLoader):
            with torch.autograd.enable_grad():
                #zero the parameter gradients
                optimizerG.zero_grad()
                optimizerD.zero_grad()
    
                #forward
                #train G and D
                x_hat = G(g_y, e_hat)
                r_hat, D_hat_res_list = D(x_hat, g_y, i=0)
                with torch.no_grad():
                    r, D_res_list = D(x, g_y, i=0)
    
                lossG = criterionG(x, x_hat, r_hat, D_res_list, D_hat_res_list)
    
                lossG.backward(retain_graph=False)
                optimizerG.step()
                
                
                #train D
                optimizerD.zero_grad()
                x_hat.detach_().requires_grad_()
                r_hat, D_hat_res_list = D(x_hat, g_y, i=0)
                r, D_res_list = D(x, g_y, i=0)
    
                lossDfake = criterionDfake(r_hat)
                lossDreal = criterionDreal(r)
    
                lossD = lossDreal + lossDfake
                lossD.backward(retain_graph=False)
                optimizerD.step()
                
                
                #train D again
                optimizerG.zero_grad()
                optimizerD.zero_grad()
                r_hat, D_hat_res_list = D(x_hat, g_y, i=0)
                r, D_res_list = D(x, g_y, i=0)
    
                lossDfake = criterionDfake(r_hat)
                lossDreal = criterionDreal(r)
    
                lossD = lossDreal + lossDfake
                lossD.backward(retain_graph=False)
                optimizerD.step()
    
    
            # Output training stats
            if epoch % 10 == 0:
                batch_end = datetime.now()
                avg_time = (batch_end - batch_start) / 10
                print('\n\navg batch time for batch size of', x.shape[0],':',avg_time)
                
                batch_start = datetime.now()
                
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(y)): %.4f'
                      % (epoch, num_epochs, i_batch, len(dataLoader),
                         lossD.item(), lossG.item(), r.mean(), r_hat.mean()))
                """
                plt.clf()
                out = x_hat.transpose(1,3)[0]
                for img_no in range(1,x_hat.shape[0]):
                    out = torch.cat((out, x_hat.transpose(1,3)[img_no]), dim = 1)
                out = out.type(torch.int32).to(cpu).numpy()*255
                plt.imshow(out)
                plt.show()
    
                plt.clf()
                out = x.transpose(1,3)[0]
                for img_no in range(1,x.shape[0]):
                    out = torch.cat((out, x.transpose(1,3)[img_no]), dim = 1)
                out = out.type(torch.int32).to(cpu).numpy()*255
                plt.imshow(out)
                plt.show()
    
                plt.clf()
                out = g_y.transpose(1,3)[0]
                for img_no in range(1,g_y.shape[0]):
                    out = torch.cat((out, g_y.transpose(1,3)[img_no]), dim = 1)
                out = out.type(torch.int32).to(cpu).numpy()*255
                plt.imshow(out)
                plt.show()
    
            lossesD.append(lossD.item())
            lossesG.append(lossG.item())"""
            if display_training:
                    plt.clf()
                    out = (x_hat[0]*255).transpose(0,2)
                    for img_no in range(1,x_hat.shape[0]):
                        out = torch.cat((out, (x_hat[img_no]*255).transpose(0,2)), dim = 1)
                    out = out.type(torch.int32).to(cpu).numpy()
                    fig = out
    
                    plt.clf()
                    out = (x[0]*255).transpose(0,2)
                    for img_no in range(1,x.shape[0]):
                        out = torch.cat((out, (x[img_no]*255).transpose(0,2)), dim = 1)
                    out = out.type(torch.int32).to(cpu).numpy()
                    fig = np.concatenate((fig, out), 0)
    
                    plt.clf()
                    out = (g_y[0]*255).transpose(0,2)
                    for img_no in range(1,g_y.shape[0]):
                        out = torch.cat((out, (g_y[img_no]*255).transpose(0,2)), dim = 1)
                    out = out.type(torch.int32).to(cpu).numpy()
                    
                    fig = np.concatenate((fig, out), 0)
                    plt.imshow(fig)
                    plt.xticks([])
                    plt.yticks([])
                    plt.draw()
                    plt.pause(0.001)
        
    num_epochs = int(input('Num epoch further?\n'))
    cont = num_epochs != 0

print('Saving finetuned model...')
torch.save({
        'epoch': epoch,
        'lossesG': lossesG,
        'lossesD': lossesD,
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        }, path_to_save)
print('...Done saving latest')

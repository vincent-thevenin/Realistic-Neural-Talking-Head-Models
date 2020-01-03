import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResBlockDown, SelfAttention, ResBlock, ResBlockD, ResBlockUp, Padding
import math
import sys

#components
class Embedder(nn.Module):
    def __init__(self, in_height):
        super(Embedder, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        
        #in 6*224*224
        self.pad = Padding(in_height) #out 6*256*256
        self.resDown1 = ResBlockDown(6, 64) #out 64*128*128
        self.resDown2 = ResBlockDown(64, 128) #out 128*64*64
        self.resDown3 = ResBlockDown(128, 256) #out 256*32*32
        self.self_att = SelfAttention(256) #out 256*32*32
        self.resDown4 = ResBlockDown(256, 512) #out 515*16*16
        self.resDown5 = ResBlockDown(512, 512) #out 512*8*8
        self.resDown6 = ResBlockDown(512, 512) #out 512*4*4
        self.sum_pooling = nn.AdaptiveMaxPool2d((1,1)) #out 512*1*1

    def forward(self, x, y):
        
        out = torch.cat((x,y),dim = -3) #out 6*224*224
        out = self.pad(out) #out 6*256*256
        out = self.resDown1(out) #out 64*128*128
        out = self.resDown2(out) #out 128*64*64
        out = self.resDown3(out) #out 256*32*32
        
        out = self.self_att(out) #out 256*32*32
        
        out = self.resDown4(out) #out 512*16*16
        out = self.resDown5(out) #out 512*8*8
        out = self.resDown6(out) #out 512*4*4
        
        out = self.sum_pooling(out) #out 512*1*1
        out = self.relu(out) #out 512*1*1
        out = out.view(-1,512,1) #out B*512*1
        return out

class Generator(nn.Module):
    P_LEN = 2*(512*2*5 + 512*2 + 512*2+ 512+256 + 256+128 + 128+64 + 64+3)
    slice_idx = [0,
                512*4, #res1
                512*4, #res2
                512*4, #res3
                512*4, #res4
                512*4, #res5
                512*4, #resUp1
                512*4, #resUp2
                512*2 + 256*2, #resUp3
                256*2 + 128*2, #resUp4
                128*2 + 64*2, #resUp5
                64*2 + 3*2] #resUp6
    for i in range(1, len(slice_idx)):
        slice_idx[i] = slice_idx[i-1] + slice_idx[i]
    
    def __init__(self, in_height, finetuning=False, e_finetuning=None):
        super(Generator, self).__init__()
        
        self.sigmoid = nn.Sigmoid()
        
        #in 3*224*224 for voxceleb2
        self.pad = Padding(in_height) #out 3*256*256
        
        #Down
        self.resDown1 = ResBlockDown(3, 64, conv_size=9, padding_size=4) #out 64*128*128
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        
        self.resDown2 = ResBlockDown(64, 128) #out 128*64*64
        self.in2 = nn.InstanceNorm2d(128, affine=True)
        
        self.resDown3 = ResBlockDown(128, 256) #out 256*32*32
        self.in3 = nn.InstanceNorm2d(256, affine=True)
        
        self.self_att_Down = SelfAttention(256) #out 256*32*32
        
        self.resDown4 = ResBlockDown(256, 512) #out 512*16*16
        self.in4 = nn.InstanceNorm2d(512, affine=True)
        
        self.resDown5 = ResBlockDown(512, 512) #out 512*8*8
        self.in5 = nn.InstanceNorm2d(512, affine=True)
        
        self.resDown6 = ResBlockDown(512, 512) #out 512*4*4
        self.in6 = nn.InstanceNorm2d(512, affine=True)
        
        #Res
        #in 512*4*4
        self.res1 = ResBlock(512)
        self.res2 = ResBlock(512)
        self.res3 = ResBlock(512)
        self.res4 = ResBlock(512)
        self.res5 = ResBlock(512)
        #out 512*4*4
        
        #Up
        #in 512*4*4
        self.resUp1 = ResBlockUp(512, 512) #out 512*8*8
        self.resUp2 = ResBlockUp(512, 512) #out 512*16*16
        self.resUp3 = ResBlockUp(512, 256) #out 256*32*32
        self.resUp4 = ResBlockUp(256, 128) #out 128*64*64
        
        self.self_att_Up = SelfAttention(128) #out 128*64*64
        
        self.resUp5 = ResBlockUp(128, 64)  #out 64*128*128
        self.resUp6 = ResBlockUp(64, 3, out_size=(in_height, in_height), scale=None, conv_size=9, padding_size=4) #out 3*224*224
        
        self.p = nn.Parameter(torch.rand(self.P_LEN,512).normal_(0.0,0.02))
        
        self.finetuning = finetuning
        self.psi = nn.Parameter(torch.rand(self.P_LEN,1))
        self.e_finetuning = e_finetuning
        
    def finetuning_init(self):
        if self.finetuning:
            self.psi = nn.Parameter(torch.mm(self.p, self.e_finetuning.mean(dim=0)))
            
    def forward(self, y, e):
        if math.isnan(self.p[0,0]):
            sys.exit()
        
        if self.finetuning:
            e_psi = self.psi.unsqueeze(0)
            e_psi = e_psi.expand(e.shape[0],self.P_LEN,1)
        else:
            p = self.p.unsqueeze(0)
            p = p.expand(e.shape[0],self.P_LEN,512)
            e_psi = torch.bmm(p, e)
        
        #in 3*224*224 for voxceleb2
        out = self.pad(y)
        
        #Encoding
        out = self.resDown1(out)
        out = self.in1(out)
        
        out = self.resDown2(out)
        out = self.in2(out)
        
        out = self.resDown3(out)
        out = self.in3(out)
        
        out = self.self_att_Down(out)
        
        out = self.resDown4(out)
        out = self.in4(out)
        
        out = self.resDown5(out)
        out = self.in5(out)
        
        out = self.resDown6(out)
        out = self.in6(out)
        
        
        #Residual
        out = self.res1(out, e_psi[:, self.slice_idx[0]:self.slice_idx[1], :])
        out = self.res2(out, e_psi[:, self.slice_idx[1]:self.slice_idx[2], :])
        out = self.res3(out, e_psi[:, self.slice_idx[2]:self.slice_idx[3], :])
        out = self.res4(out, e_psi[:, self.slice_idx[3]:self.slice_idx[4], :])
        out = self.res5(out, e_psi[:, self.slice_idx[4]:self.slice_idx[5], :])
        
        
        #Decoding
        out = self.resUp1(out, e_psi[:, self.slice_idx[5]:self.slice_idx[6], :])
        
        out = self.resUp2(out, e_psi[:, self.slice_idx[6]:self.slice_idx[7], :])
        
        out = self.resUp3(out, e_psi[:, self.slice_idx[7]:self.slice_idx[8], :])
        
        out = self.resUp4(out, e_psi[:, self.slice_idx[8]:self.slice_idx[9], :])
        
        out = self.self_att_Up(out)
        
        out = self.resUp5(out, e_psi[:, self.slice_idx[9]:self.slice_idx[10], :])
        
        out = self.resUp6(out, e_psi[:, self.slice_idx[10]:self.slice_idx[11], :])
        out = self.sigmoid(out)
        
        out = out*255
        
        #out 3*224*224
        return out
        
class Discriminator(nn.Module):
    def __init__(self, num_videos, finetuning=False, e_finetuning=None):
        super(Discriminator, self).__init__()
        
        #in 6*224*224
        self.pad = Padding(224) #out 6*256*256
        self.resDown1 = ResBlockDown(6, 64) #out 64*128*128
        self.resDown2 = ResBlockDown(64, 128) #out 128*64*64
        self.resDown3 = ResBlockDown(128, 256) #out 256*32*32
        self.self_att = SelfAttention(256) #out 256*32*32
        self.resDown4 = ResBlockDown(256, 512) #out 512*16*16
        self.resDown5 = ResBlockDown(512, 512) #out 512*8*8
        self.resDown6 = ResBlockDown(512, 512) #out 512*4*4
        self.res = ResBlockD(512) #out 512*4*4
        self.sum_pooling = nn.AdaptiveAvgPool2d((1,1)) #out 512*1*1
        
        self.W_i = nn.Parameter(torch.rand(512, num_videos))
        self.w_0 = nn.Parameter(torch.randn(512,1))
        self.b = nn.Parameter(torch.randn(1))
        
        self.finetuning = finetuning
        self.e_finetuning = e_finetuning
        self.w_prime = nn.Parameter( torch.randn(512,1) )
        
    def finetuning_init(self):
        if self.finetuning:
            self.w_prime = nn.Parameter( self.w_0 + self.e_finetuning.mean(dim=0))
        
    def forward(self, x, y, i):
        out = torch.cat((x,y), dim=-3) #out B*6*224*224
        
        out = self.pad(out)
        
        out1 = self.resDown1(out)
        
        out2 = self.resDown2(out1)
        
        out3 = self.resDown3(out2)
        
        out = self.self_att(out3)
        
        out4 = self.resDown4(out)
        
        out5 = self.resDown5(out4)
        
        out6 = self.resDown6(out5)
        
        out7 = self.res(out6)
        
        out = self.sum_pooling(out7)
        
        out = out.view(-1,512,1) #out B*512*1
        
        if self.finetuning:
            out = torch.bmm(out.transpose(1,2), (self.w_prime.unsqueeze(0).expand(out.shape[0],512,1))) + self.b
        else:
            out = torch.bmm(out.transpose(1,2), (self.W_i[:,i].unsqueeze(-1)).transpose(0,1) + self.w_0) + self.b #1x1
        
        return out, [out1 , out2, out3, out4, out5, out6, out7]

class Cropped_VGG19(nn.Module):
    def __init__(self):
        super(Cropped_VGG19, self).__init__()
        
        self.conv1_1 = nn.Conv2d(3,64,3)
        self.conv1_2 = nn.Conv2d(64,64,3)
        self.conv2_1 = nn.Conv2d(64,128,3)
        self.conv2_2 = nn.Conv2d(128,128,3)
        self.conv3_1 = nn.Conv2d(128,256,3)
        self.conv3_2 = nn.Conv2d(256,256,3)
        self.conv3_3 = nn.Conv2d(256,256,3)
        self.conv4_1 = nn.Conv2d(256,512,3)
        self.conv4_2 = nn.Conv2d(512,512,3)
        self.conv4_3 = nn.Conv2d(512,512,3)
        self.conv5_1 = nn.Conv2d(512,512,3)
        #self.conv5_2 = nn.Conv2d(512,512,3)
        #self.conv5_3 = nn.Conv2d(512,512,3)
        
    def forward(self, x):
        conv1_1_pad     = F.pad(x, (1, 1, 1, 1))
        conv1_1         = self.conv1_1(conv1_1_pad)
        relu1_1         = F.relu(conv1_1)
        conv1_2_pad     = F.pad(relu1_1, (1, 1, 1, 1))
        conv1_2         = self.conv1_2(conv1_2_pad)
        relu1_2         = F.relu(conv1_2)
        pool1_pad       = F.pad(relu1_2, (0, 1, 0, 1), value=float('-inf'))
        pool1           = F.max_pool2d(pool1_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv2_1_pad     = F.pad(pool1, (1, 1, 1, 1))
        conv2_1         = self.conv2_1(conv2_1_pad)
        relu2_1         = F.relu(conv2_1)
        conv2_2_pad     = F.pad(relu2_1, (1, 1, 1, 1))
        conv2_2         = self.conv2_2(conv2_2_pad)
        relu2_2         = F.relu(conv2_2)
        pool2_pad       = F.pad(relu2_2, (0, 1, 0, 1), value=float('-inf'))
        pool2           = F.max_pool2d(pool2_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv3_1_pad     = F.pad(pool2, (1, 1, 1, 1))
        conv3_1         = self.conv3_1(conv3_1_pad)
        relu3_1         = F.relu(conv3_1)
        conv3_2_pad     = F.pad(relu3_1, (1, 1, 1, 1))
        conv3_2         = self.conv3_2(conv3_2_pad)
        relu3_2         = F.relu(conv3_2)
        conv3_3_pad     = F.pad(relu3_2, (1, 1, 1, 1))
        conv3_3         = self.conv3_3(conv3_3_pad)
        relu3_3         = F.relu(conv3_3)
        pool3_pad       = F.pad(relu3_3, (0, 1, 0, 1), value=float('-inf'))
        pool3           = F.max_pool2d(pool3_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv4_1_pad     = F.pad(pool3, (1, 1, 1, 1))
        conv4_1         = self.conv4_1(conv4_1_pad)
        relu4_1         = F.relu(conv4_1)
        conv4_2_pad     = F.pad(relu4_1, (1, 1, 1, 1))
        conv4_2         = self.conv4_2(conv4_2_pad)
        relu4_2         = F.relu(conv4_2)
        conv4_3_pad     = F.pad(relu4_2, (1, 1, 1, 1))
        conv4_3         = self.conv4_3(conv4_3_pad)
        relu4_3         = F.relu(conv4_3)
        pool4_pad       = F.pad(relu4_3, (0, 1, 0, 1), value=float('-inf'))
        pool4           = F.max_pool2d(pool4_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv5_1_pad     = F.pad(pool4, (1, 1, 1, 1))
        conv5_1         = self.conv5_1(conv5_1_pad)
        
        return [conv1_1, conv2_1, conv3_1, conv4_1, conv5_1]
    
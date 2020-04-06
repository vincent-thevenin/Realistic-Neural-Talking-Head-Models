import torch
import cv2
from matplotlib import pyplot as plt

from loss.loss_discriminator import *
from loss.loss_generator import *
from network.blocks import *
from network.model import *
from webcam_demo.webcam_extraction_conversion import *

from params.params import path_to_chkpt
from tqdm import tqdm

"""Init"""

#Paths
path_to_model_weights = 'finetuned_model.tar'
path_to_embedding = 'e_hat_video.tar'
path_to_mp4 = 'test_vid2.webm'

device = torch.device("cuda:0")
cpu = torch.device("cpu")

checkpoint = torch.load(path_to_model_weights, map_location=cpu) 
e_hat = torch.load(path_to_embedding, map_location=cpu)
e_hat = e_hat['e_hat'].to(device)

G = Generator(256, finetuning=True, e_finetuning=e_hat)
G.eval()

"""Training Init"""
G.load_state_dict(checkpoint['G_state_dict'])
G.to(device)


"""Main"""
print('PRESS Q TO EXIT')
cap = cv2.VideoCapture(path_to_mp4)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
ret = True
i = 0
size = (256*3,256)
#out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc('M','P','4','2'), 30, size)
video = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

with torch.no_grad():
    while ret:
        x, g_y, ret = generate_landmarks(cap=cap, device=device, pad=50)
        if ret:
            g_y = g_y.unsqueeze(0)/255
            x = x.unsqueeze(0)/255


            #forward
            # Calculate average encoding vector for video
            #f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]) #BxK,2,3,224,224
            #train G

            x_hat = G(g_y, e_hat)

            plt.clf()
            out1 = x_hat.transpose(1,3)[0]
            #for img_no in range(1,x_hat.shape[0]):
            #    out1 = torch.cat((out1, x_hat.transpose(1,3)[img_no]), dim = 1)
            out1 = out1.to(cpu).numpy()
            #plt.imshow(out1)
            #plt.show()

            #plt.clf()
            out2 = x.transpose(1,3)[0]
            #for img_no in range(1,x.shape[0]):
            #    out2 = torch.cat((out2, x.transpose(1,3)[img_no]), dim = 1)
            out2 = out2.to(cpu).numpy()
            #plt.imshow(out2)
            #plt.show()

            #plt.clf()
            out3 = g_y.transpose(1,3)[0]
            #for img_no in range(1,g_y.shape[0]):
            #    out3 = torch.cat((out3, g_y.transpose(1,3)[img_no]), dim = 1)
            out3 = out3.to(cpu).numpy()
            #plt.imshow(out3)
            #plt.show()

            fake = cv2.cvtColor(out1*255, cv2.COLOR_BGR2RGB)
            me = cv2.cvtColor(out2*255, cv2.COLOR_BGR2RGB)
            landmark = cv2.cvtColor(out3*255, cv2.COLOR_BGR2RGB)
            img = np.concatenate((me, landmark, fake), axis=1)
            img = img.astype('uint8')
            video.write(img)

            i+=1
            print(i,'/',n_frames)
cap.release()
video.release()
"""cv2.destroyAllWindows()"""
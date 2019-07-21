"""Main"""
import torch
from matplotlib import pyplot as plt
import os
import cv2

from dataset.video_extraction_conversion import select_frames
from network.blocks import *
from network.model import Embedder

from webcam_demo.webcam_extraction_conversion import get_borders, crop_and_reshape_preds, crop_and_reshape_img
import face_alignment
import numpy as np


"""Create dataset and net"""
device = torch.device("cuda:0")
cpu = torch.device("cpu")
path_to_e_hat_video = 'ehat_video.tar'
path_to_e_hat_images = 'ehat_images.tar'
path_to_chkpt = 'model_weights.tar'
path_to_video = 'examples/fine_tuning/test_video.mp4'
path_to_images = 'examples/fine_tuning/test_images'
T = 32


def select_images_frames(path_to_images):
    images_list = []
    for image_name in os.listdir(path_to_images):
        img = cv2.imread(os.path.join(path_to_images, image_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images_list.append(img)
    return images_list

def generate_cropped_landmarks(frames_list, pad=50):
    frame_landmark_list = []
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device ='cuda:0')
    
    for i in range(len(frames_list)):
        try:
            input = frames_list[i]
            preds = fa.get_landmarks(input)[0]
            
            input = crop_and_reshape_img(input, preds, pad=pad)
            preds = crop_and_reshape_preds(preds, pad=pad)

            dpi = 100
            fig = plt.figure(figsize=(input.shape[1]/dpi, input.shape[0]/dpi), dpi = dpi)
            ax = fig.add_subplot(1,1,1)
            ax.imshow(np.ones(input.shape))
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            #chin
            ax.plot(preds[0:17,0],preds[0:17,1],marker='',markersize=5,linestyle='-',color='green',lw=2)
            #left and right eyebrow
            ax.plot(preds[17:22,0],preds[17:22,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
            ax.plot(preds[22:27,0],preds[22:27,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
            #nose
            ax.plot(preds[27:31,0],preds[27:31,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
            ax.plot(preds[31:36,0],preds[31:36,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
            #left and right eye
            ax.plot(preds[36:42,0],preds[36:42,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
            ax.plot(preds[42:48,0],preds[42:48,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
            #outer and inner lip
            ax.plot(preds[48:60,0],preds[48:60,1],marker='',markersize=5,linestyle='-',color='purple',lw=2)
            ax.plot(preds[60:68,0],preds[60:68,1],marker='',markersize=5,linestyle='-',color='pink',lw=2) 
            ax.axis('off')

            fig.canvas.draw()

            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            frame_landmark_list.append((input, data))
            plt.close(fig)
        except:
            print('Error: Video corrupted or no landmarks visible')
    
    for i in range(len(frames_list) - len(frame_landmark_list)):
        #filling frame_landmark_list in case of error
        frame_landmark_list.append(frame_landmark_list[i])
    
    
    return frame_landmark_list




"""Loading Embedder input"""

frame_mark_video = select_frames(path_to_video , T)
frame_mark_video = generate_cropped_landmarks(frame_mark_video, pad=50)
frame_mark_video = torch.from_numpy(np.array(frame_mark_video)).type(dtype = torch.float) #T,2,256,256,3
frame_mark_video = frame_mark_video.transpose(2,4).to(device) #T,2,3,256,256
f_lm_video = frame_mark_video.unsqueeze(0) #1,T,2,3,256,256

frame_mark_images = select_images_frames(path_to_images)
frame_mark_images = generate_cropped_landmarks(frame_mark_images, pad=50)
frame_mark_images = torch.from_numpy(np.array(frame_mark_images)).type(dtype = torch.float) #T,2,256,256,3
frame_mark_images = frame_mark_images.transpose(2,4).to(device) #T,2,3,256,256
f_lm_images = frame_mark_images.unsqueeze(0) #1,T,2,3,256,256



E = Embedder(256).to(device)
E.eval()



"""Loading from past checkpoint"""
checkpoint = torch.load(path_to_chkpt, map_location=cpu)
E.load_state_dict(checkpoint['E_state_dict'])




"""Inference"""
with torch.no_grad():
    #forward
    # Calculate average encoding vector for video
    f_lm = f_lm_video
    f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]) #BxT,2,3,224,224
    e_vectors = E(f_lm_compact[:,0,:,:,:], f_lm_compact[:,1,:,:,:]) #BxT,512,1
    e_vectors = e_vectors.view(-1, f_lm.shape[1], 512, 1) #B,T,512,1
    e_hat_video = e_vectors.mean(dim=1)
    
    
    f_lm = f_lm_images
    f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]) #BxT,2,3,224,224
    e_vectors = E(f_lm_compact[:,0,:,:,:], f_lm_compact[:,1,:,:,:]) #BxT,512,1
    e_vectors = e_vectors.view(-1, f_lm.shape[1], 512, 1) #B,T,512,1
    e_hat_images = e_vectors.mean(dim=1)


print('Saving e_hat...')
torch.save({
        'e_hat': e_hat_video
        }, path_to_e_hat_video)
torch.save({
		'e_hat': e_hat_images
		}, path_to_e_hat_images)
print('...Done saving')
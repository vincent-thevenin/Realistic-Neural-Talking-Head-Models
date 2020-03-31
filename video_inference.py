import torch
import os
from datetime import datetime
import numpy as np
import cv2
# from torchvision.utils import save_image
from tqdm import tqdm
import face_alignment
from matplotlib import pyplot as plt

path_to_mp4 = 'test_vid.mp4'
K = 8
num_vid = 0
device = torch.device('cuda:0')
saves_dir = 'vid_saves'
face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device ='cuda:0')

if not os.path.isdir(saves_dir):
    os.mkdir(saves_dir)


def get_borders(preds):
    minX = maxX = preds[0,0]
    minY = maxY = preds[0,1]
    
    for i in range(1, len(preds)):
        x = preds[i,0]
        if x < minX:
            minX = x
        elif x > maxX:
            maxX = x
        
        y = preds[i,1]
        if y < minY:
            minY = y
        elif y > maxY:
            maxY = y
    
    return minX, maxX, minY, maxY

def crop_and_reshape_preds(preds, pad, out_shape=256):
    minX, maxX, minY, maxY = get_borders(preds)
    
    delta = max(maxX - minX, maxY - minY)
    deltaX = (delta - (maxX - minX))/2
    deltaY = (delta - (maxY - minY))/2
    
    deltaX = int(deltaX)
    deltaY = int(deltaY)
    
    
    #crop
    for i in range(len(preds)):
        preds[i][0] = max(0, preds[i][0] - minX + deltaX + pad)
        preds[i][1] = max(0, preds[i][1] - minY + deltaY + pad)
    
    #find reshape factor
    r = out_shape/(delta + 2*pad)
        
    for i in range(len(preds)):
        preds[i,0] = int(r*preds[i,0])
        preds[i,1] = int(r*preds[i,1])
    return preds

def crop_and_reshape_img(img, preds, pad, out_shape=256):
    minX, maxX, minY, maxY = get_borders(preds)
    
    #find reshape factor
    delta = max(maxX - minX, maxY - minY)
    deltaX = (delta - (maxX - minX))/2
    deltaY = (delta - (maxY - minY))/2
    
    minX = int(minX)
    maxX = int(maxX)
    minY = int(minY)
    maxY = int(maxY)
    deltaX = int(deltaX)
    deltaY = int(deltaY)
    
    lowY = max(0,minY-deltaY-pad)
    lowX = max(0, minX-deltaX-pad)
    img = img[lowY:maxY+deltaY+pad, lowX:maxX+deltaX+pad, :]
    img = cv2.resize(img, (out_shape,out_shape))
    
    return img

def generate_cropped_landmarks(frames_list, face_aligner, pad=50):
    frame_landmark_list = []
    fa = face_aligner
    
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

def pick_images(video_path, num_images, cap, n_frames, idx):    
    #idxes = [int(i>=idx and i<(idx+n_frames)) for i in range(n_frames)]
    
    frames_list = []
    
    # Read until video is completed or no frames needed
    ret = True
    frame_counter = 0
    for i in range(num_images):
        ret, frame = cap.read()
        
        if ret:
            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append(RGB)
    
    return frames_list
    

cap = cv2.VideoCapture(path_to_mp4)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_path = path_to_mp4
""" for person_id in tqdm(os.listdir(path_to_mp4)):
    for video_id in tqdm(os.listdir(os.path.join(path_to_mp4, person_id))):
        for video in os.listdir(os.path.join(path_to_mp4, person_id, video_id)):
"""
for img_idx in range(n_frames//K):
    frame_mark = pick_images(video_path, K, cap, n_frames, img_idx*K)
    frame_mark = generate_cropped_landmarks(frame_mark, face_aligner)
    if len(frame_mark) == K:
        final_list = [frame_mark[i][0] for i in range(K)]
        for i in range(K):
            final_list.append(frame_mark[i][1]) #K*2,224,224,3
        final_list = np.array(final_list)
        final_list = np.transpose(final_list, [1,0,2,3])
        final_list = np.reshape(final_list, (256, 256*2*K, 3))
        final_list = cv2.cvtColor(final_list, cv2.COLOR_BGR2RGB)
        
        if not os.path.isdir(saves_dir):
            os.mkdir(saves_dir)
            
        cv2.imwrite(saves_dir+"/"+str(img_idx)+".png", final_list)
        num_vid += 1

            
            
cap.release()
print('done')

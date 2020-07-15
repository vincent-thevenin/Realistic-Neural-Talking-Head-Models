import torch
import os
from datetime import datetime
import numpy as np
import cv2
# from torchvision.utils import save_image
from tqdm import tqdm
import face_alignment
from matplotlib import pyplot as plt
from .params.params import path_to_mp4, path_to_preprocess

K = 8
num_vid = 0
device = torch.device('cuda:0')
face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device ='cuda:0')

if not os.path.isdir(path_to_preprocess):
    os.mkdir(path_to_preprocess)

def generate_landmarks(frames_list, face_aligner):
    frame_landmark_list = []
    fa = face_aligner
    
    for i in range(len(frames_list)):
        try:
            input = frames_list[i]
            preds = fa.get_landmarks(input)[0]

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

def pick_images(video_path, pic_folder, num_images):
    cap = cv2.VideoCapture(video_path)
    
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    idxes = [1 if i%(n_frames//num_images+1)==0 else 0 for i in range(n_frames)]
    
    frames_list = []
    
    # Read until video is completed or no frames needed
    ret = True
    frame_idx = 0
    frame_counter = 0
    while(ret and frame_idx < n_frames):
        ret, frame = cap.read()
        
        if ret and idxes[frame_idx] == 1:
            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append(RGB)
            
            # frame_counter += 1
            # pic_path = pic_folder + '_'+str(frame_counter)+'.jpg'
            # cv2.imwrite(pic_path, frame)
            
        frame_idx += 1

    cap.release()
    
    return frames_list
    

for person_id in tqdm(os.listdir(path_to_mp4)):
    for video_id in tqdm(os.listdir(os.path.join(path_to_mp4, person_id))):
        for video in os.listdir(os.path.join(path_to_mp4, person_id, video_id)):
            
                
            try:
                video_path = os.path.join(path_to_mp4, person_id, video_id, video)
                frame_mark = pick_images(video_path, path_to_preprocess+'/' + person_id+'/'+video_id+'/'+video.split('.')[0], K)
                frame_mark = generate_landmarks(frame_mark, face_aligner)
                if len(frame_mark) == K:
                    final_list = [frame_mark[i][0] for i in range(K)]
                    for i in range(K):
                        final_list.append(frame_mark[i][1]) #K*2,224,224,3
                    final_list = np.array(final_list)
                    final_list = np.transpose(final_list, [1,0,2,3])
                    final_list = np.reshape(final_list, (224, 224*2*K, 3))
                    final_list = cv2.cvtColor(final_list, cv2.COLOR_BGR2RGB)
                    
                    if not os.path.isdir(path_to_preprocess+"/"+str(num_vid//256)):
                        os.mkdir(path_to_preprocess+"/"+str(num_vid//256))
                        
                    cv2.imwrite(path_to_preprocess+"/"+str(num_vid//256)+"/"+str(num_vid)+".png", final_list)
                    num_vid += 1
                    break #take only one video

                    
            except:
                print('ERROR: ', video_path)
            
        
print('done')

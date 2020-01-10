from video_extraction_conversion import *
import torch
import os
from datetime import datetime
import numpy as np
import cv2
from torchvision.utils import save_image
from tqdm import tqdm

path_to_mp4 = '/mnt/ACA21355A21322FE/VoxCeleb/vox2_mp4/dev/mp4'
K = 8
device = torch.device('cuda:0')
saves_dir = '/mnt/ACA21355A21322FE/VoxCeleb/saves'
face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device ='cuda:0')

if not os.path.isdir(saves_dir):
    os.mkdir(saves_dir)

def save_images(video_path, pic_folder, num_images):
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
            
            frame_counter += 1
            pic_path = pic_folder + '_'+str(frame_counter)+'.jpg'
            cv2.imwrite(pic_path, frame)
            
        frame_idx += 1

    cap.release()
    
    return frames_list
    
    
progress = 0
total = 145568
mean = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(0).expand(K*2,3).reshape(-1).to(device)
std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(0).expand(K*2,3).reshape(-1).to(device)
elapsed_time = datetime.now()   
elapsed_time -= elapsed_time
timer_start = datetime.now()
for person_id in tqdm(os.listdir(path_to_mp4)):
    for video_id in os.listdir(os.path.join(path_to_mp4, person_id)):
        for video in os.listdir(os.path.join(path_to_mp4, person_id, video_id)):
            
            progress += 1
            
            if not os.path.isdir(saves_dir+'/' + person_id):
                os.mkdir(saves_dir+'/' + person_id)
            if not os.path.isdir(saves_dir +'/'+ person_id+'/'+video_id):
                os.mkdir(saves_dir +'/'+ person_id+'/'+video_id)
            elif len(os.listdir(saves_dir +'/'+ person_id+'/'+video_id)) == 16:
                total -= 1
                progress -= 1
                break
                
            try:
                video_path = os.path.join(path_to_mp4, person_id, video_id, video)
                frame_mark = save_images(video_path, saves_dir+'/' + person_id+'/'+video_id+'/'+video.split('.')[0], K)
                frame_mark = generate_landmarks(frame_mark, face_aligner)
                for i, frame in enumerate(frame_mark):
                    save_image(torch.from_numpy(frame[1]).type(dtype = torch.float).transpose(0,2).transpose(1,2), saves_dir+'/' + person_id+'/'+video_id+'/LM'+'_'+str(i)+'.jpg')
                # frame_mark = torch.from_numpy(np.array(frame_mark)).type(dtype = torch.float) #K,2,224,224,3
    #             frame_mark = frame_mark.transpose(2,4).to(device) #K,2,3,224,224
    #             frame_mark = frame_mark.view(-1,frame_mark.shape[-1],frame_mark.shape[-1])/255
    
    # #            frame_mark = normalize(frame_mark, mean, std, inplace=True)
    #             frame_mark = frame_mark.view(K, 2, 3, frame_mark.shape[-1], frame_mark.shape[-1])
                
    #             g_idx = torch.randint(low = 0, high = K, size = (1,1))
    #             x = frame_mark[g_idx,0].squeeze()
    #             g_y = frame_mark[g_idx,1].squeeze()
    #             torch.save({
    #                 'frame_mark': frame_mark,
    #                 'x': x,
    #                 'g_y':g_y,
    #                 }, saves_dir +'/'+ person_id+'/'+video_id+'/'+video.split('.')[0]+'.pth')
                # if progress%(total//10000+1)==1:
                #     timer_stop = datetime.now()
                #     diff_time = timer_stop - timer_start
                #     timer_start = timer_stop
                #     elapsed_time += diff_time
                #     print('ETA:',elapsed_time*(total/progress - 1),'\n')
            except:
                print('ERROR: ', video_path)
            
            break
        
print('done')
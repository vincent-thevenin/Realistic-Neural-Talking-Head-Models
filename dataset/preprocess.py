from video_extraction_conversion import *
import torch
import os
from torchvision.transforms.functional import normalize
from datetime import datetime

path_to_mp4 = '/media/vincent/disk/Realistic-Neural-Talking-Head-Models/mp4'
K = 8
device = torch.device('cuda:0')
saves_dir = '/media/vincent/LaCie 500/Realistic-Neural-Talking-Head-Models/saves/'

if not os.path.isdir(saves_dir):
    os.mkdir(saves_dir)


progress = 0
total = 315000
mean = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(0).expand(K*2,3).reshape(-1).to(device)
std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(0).expand(K*2,3).reshape(-1).to(device)
elapsed_time = datetime.now()   
elapsed_time -= elapsed_time
timer_start = datetime.now()
for person_id in os.listdir(path_to_mp4):
    for video_id in os.listdir(os.path.join(path_to_mp4, person_id)):
        for video in os.listdir(os.path.join(path_to_mp4, person_id, video_id)):
            progress += 1
            path = os.path.join(path_to_mp4, person_id, video_id, video)
            frame_mark = select_frames(path , K)
            frame_mark = generate_landmarks(frame_mark)
            frame_mark = torch.from_numpy(np.array(frame_mark)).type(dtype = torch.float) #K,2,224,224,3
            frame_mark = frame_mark.transpose(2,4).to(device) #K,2,3,224,224
            frame_mark = frame_mark.view(-1,frame_mark.shape[-1],frame_mark.shape[-1])/255

#            frame_mark = normalize(frame_mark, mean, std, inplace=True)
            frame_mark = frame_mark.view(K, 2, 3, frame_mark.shape[-1], frame_mark.shape[-1])
            
            g_idx = torch.randint(low = 0, high = K, size = (1,1))
            x = frame_mark[g_idx,0].squeeze()
            g_y = frame_mark[g_idx,1].squeeze()
            if not os.path.isdir(saves_dir + person_id):
                os.mkdir(saves_dir + person_id)
            if not os.path.isdir(saves_dir + person_id+'/'+video_id):
                os.mkdir(saves_dir + person_id+'/'+video_id)
            #torch.save({
             #   'frame_mark': frame_mark,
              #  'x': x,
               # 'g_y':g_y,
               # }, saves_dir + person_id+'/'+video_id+'/'+video.split('.')[0]+'.pth')
            if progress%(total//100000+1)==1:
                timer_stop = datetime.now()
                diff_time = timer_stop - timer_start
                time_start = timer_stop
                elapsed_time += diff_time
                print('ETA:',elapsed_time*(total/progress - 1),'\n')
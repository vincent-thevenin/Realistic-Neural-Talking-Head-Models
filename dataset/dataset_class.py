import torch
from torch.utils.data import Dataset
import os

from .video_extraction_conversion import *


class VidDataSet(Dataset):
    def __init__(self, K, path_to_mp4):
        self.K = K
        self.path_to_mp4 = path_to_mp4
    
    def __len__(self):
        vid_num = 0
        for person_id in os.listdir(self.path_to_mp4):
            for video_id in os.listdir(os.path.join(self.path_to_mp4, person_id)):
                for video in os.listdir(os.path.join(self.path_to_mp4, person_id, video_id)):
                    vid_num += 1
        return vid_num
    
    def __getitem__(self, idx):
        vid_idx = idx
        if idx<0:
            idx = self.__len__() + idx
        for person_id in os.listdir(self.path_to_mp4):
            for video_id in os.listdir(os.path.join(self.path_to_mp4, person_id)):
                for video in os.listdir(os.path.join(self.path_to_mp4, person_id, video_id)):
                    if idx != 0:
                        idx -= 1
                    else:
                        break
                if idx == 0:
                    break
            if idx == 0:
                break
        path = os.path.join(self.path_to_mp4, person_id, video_id, video)
        frame_mark = select_frames(path , self.K)
        frame_mark = generate_landmarks(frame_mark)
        frame_mark = torch.from_numpy(np.array(frame_mark)).type(dtype = torch.float) #K,2,224,224,3
        frame_mark = frame_mark.transpose(2,4).to(device) #K,2,3,224,224
        
        g_idx = torch.randint(low = 0, high = self.K, size = (1,1))
        x = frame_mark[g_idx,0].squeeze()
        g_y = frame_mark[g_idx,1].squeeze()
        return frame_mark, x, g_y, vid_idx
        
        
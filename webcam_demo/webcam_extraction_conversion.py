import cv2
import face_alignment
from matplotlib import pyplot as plt
import numpy as np
import torch

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


def generate_landmarks(cap, device, pad):
    """Input: cap a cv2.VideoCapture object, device the torch.device, 
pad the distance in pixel from border to face

output: x the camera output, g_y the corresponding landmark"""
   
    #Get webcam image
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda:0')
    no_pic = True
    
    while(no_pic == True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list = [RGB]


            #Create landmark for face
            frame_landmark_list = []

            for i in range(len(frames_list)):
                try:
                    input = frames_list[i]
                    preds = fa.get_landmarks(input)[0]

                    input = crop_and_reshape_img(input, preds, pad=pad)
                    preds = crop_and_reshape_preds(preds, pad=pad)

                    dpi = 100
                    fig = plt.figure(figsize=(256/dpi, 256/dpi), dpi = dpi)
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
                    no_pic = False
                except:
                    print('Error: Video corrupted or no landmarks visible')
        else:
            break
    if ret:
        frame_mark = torch.from_numpy(np.array(frame_landmark_list)).type(dtype = torch.float) #K,2,256,256,3
        frame_mark = frame_mark.transpose(2,4).to(device) #K,2,3,256,256

        x = frame_mark[0,0].to(device)
        g_y = frame_mark[0,1].to(device)
    else:
        x = g_y = None
    
    return x,g_y,ret

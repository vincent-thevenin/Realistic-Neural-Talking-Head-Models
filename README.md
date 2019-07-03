# Realistic-Neural-Talking-Head-Models
My implementation of Few-Shot Adversarial Learning of Realistic Neural Talking Head Models (Egor Zakharov et al.).

## Prerequisites

### 1.Loading and converting the caffe VGGFace model to pytorch for the content loss:
Follow these instructions to install the VGGFace from the paper (https://arxiv.org/pdf/1703.07332.pdf):

`wget https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f02f8769e64494bcd3d7e97d5d747ac275825721/VGG_ILSVRC_19_layers_deploy.prototxt`

`wget http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel`

`apt install caffe-cuda`

`pip install mmdnn`

Convert Caffe to IR (Intermediate Representation)

`mmtoir -f caffe -n VGG_ILSVRC_19_layers_deploy.prototxt -w VGG_ILSVRC_19_layers.caffemodel -o VGGFACE_IR`

**If you have a problem with pickle, delete your numpy and reinstall numpy with version 1.16.1**

IR to Pytorch code and weights

`mmtocode -f pytorch -n VGGFACE_IR.pb --IRWeightPath VGGFACE_IR.npy --dstModelPath Pytorch_VGGFACE_IR.py -dw Pytorch_VGGFACE_IR.npy`

Pytorch code and weights to Pytorch model

`mmtomodel -f pytorch -in Pytorch_VGGFACE_IR.py -iw Pytorch_VGGFACE_IR.npy -o Pytorch_VGGFACE.pth`


At this point, you will have a few files in your directory. To save some space you can delete everything and keep **Pytorch_VGGFACE_IR.py** and **Pytorch_VGGFACE.pth**

### 2.Libraries
- face-alignment
- torch
- numpy
- cv2 (opencv-python)
- matplotlib

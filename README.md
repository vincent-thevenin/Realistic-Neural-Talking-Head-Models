# Realistic-Neural-Talking-Head-Models
My implementation of Few-Shot Adversarial Learning of Realistic Neural Talking Head Models (Egor Zakharov et al.). https://arxiv.org/pdf/1905.08233.pdf

## Prerequisites

### 1.Loading and converting the caffe VGGFace model to pytorch for the content loss:
Follow these instructions to install the VGGFace from the paper (https://arxiv.org/pdf/1703.07332.pdf):

```
$ wget https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f02f8769e64494bcd3d7e97d5d747ac275825721/VGG_ILSVRC_19_layers_deploy.prototxt
$ wget http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel
$ sudo apt install caffe-cuda
$ pip install mmdnn
```

Convert Caffe to IR (Intermediate Representation)

`$ mmtoir -f caffe -n VGG_ILSVRC_19_layers_deploy.prototxt -w VGG_ILSVRC_19_layers.caffemodel -o VGGFACE_IR`

**If you have a problem with pickle, delete your numpy and reinstall numpy with version 1.16.1**

IR to Pytorch code and weights

`$ mmtocode -f pytorch -n VGGFACE_IR.pb --IRWeightPath VGGFACE_IR.npy --dstModelPath Pytorch_VGGFACE_IR.py -dw Pytorch_VGGFACE_IR.npy`

Pytorch code and weights to Pytorch model

`$ mmtomodel -f pytorch -in Pytorch_VGGFACE_IR.py -iw Pytorch_VGGFACE_IR.npy -o Pytorch_VGGFACE.pth`


At this point, you will have a few files in your directory. To save some space you can delete everything and keep **Pytorch_VGGFACE_IR.py** and **Pytorch_VGGFACE.pth**

### 2.Libraries
- face-alignment
- torch
- numpy
- cv2 (opencv-python)
- matplotlib

### 3.VoxCeleb2 Dataset
The VoxCeleb2 dataset has videos in zip format. (Very heavy 270GB for the dev one and 8GB for the test)
http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html

## Architecture

I followed the architecture guidelines from the paper on top of details provided by M. Zakharov.

The images that are fed from voxceleb2 are resized from 224x224 to 256x256 by using zero-padding. This is done so that spatial dimensions don't get rounded when passing through downsampling layers.

The residuals blocks are from [LARGE SCALE GAN TRAINING FOR HIGH FIDELITY NATURAL IMAGE SYNTHESIS](https://arxiv.org/pdf/1809.11096.pdf)

**Embedder**

The embedder uses 6 downsampling residual blocks with no normalisation. A self-attention layer is added in the middle.
The output from the last residual block is resized to a vector of size 512 via maxpooling.

**Generator**

The downsampling part of the generator uses the same architecture as the embedder with instance normalization added at each block following the paper.

The same dimension residual part uses 5 blocks. These blocks use adaptive instance normalization. Unlike the [AdaIN paper](https://arxiv.org/pdf/1703.06868.pdf) where the alpha and beta learnable parameters from instance normalisation are replaced with mean and variance of the input style, the adaptative parameters (mean and variance) are taken from psi. With psi = P\*e, P the projection matrix and e the embedding vector calculated by the embedder.

(*P is of size 2\*(512\*2\*5 + 512\*2 + 512\*2+ 512+256 + 256+128 + 128+64 + 64+3) x 512 = 17158 x 512*)

There are then 6 upsampling residual blocks. The final output is a tensor of dimensions 3x224x224. I rescale the image using a sigmoid and multiplying by 255.

Self-attention layers are added both in the downsampling part and upsampling part of the generator.

**Discriminator**

The discriminator uses the same architecture as the embedder.

# Realistic-Neural-Talking-Head-Models
Implementation of Few-Shot Adversarial Learning of Realistic Neural Talking Head Models (Egor Zakharov et al.). https://arxiv.org/abs/1905.08233

This repo is based on https://github.com/vincent-thevenin/Realistic-Neural-Talking-Head-Models

### My changes to the original repo
Download [caffe-trained version of VGG19 converted to pytorch ](https://web.eecs.umich.edu/~justincj/models/vgg19-d01eb7cb.pth).

As there are some layer names mismatching in the converted model, 

change `VGG19_caffe_weight_path` in params.py to your path and run
```
python change_vgg19_caffelayer_name.py
```


Main code changes in `loss_generator.py`:
```
self.vgg19_caffe_RGB_mean = torch.FloatTensor([123.68, 116.779, 103.939]).view(1, 3, 1, 1).to(device) # RGB order
self.vggface_caffe_RGB_mean = torch.FloatTensor([129.1863,104.7624,93.5940]).view(1, 3, 1, 1).to(device) # RGB order

x_vgg19 = x * 255  - self.vgg19_caffe_RGB_mean
x_vgg19 = x_vgg19[:,[2,1,0],:,:]
x_hat_vgg19 = x_hat * 255 - self.vgg19_caffe_RGB_mean
x_hat_vgg19 = x_hat_vgg19[:,[2,1,0],:,:]
x_vggface = x * 255 - self.vggface_caffe_RGB_mean
x_vggface = x_vggface[:,[2,1,0],:,:] # B RGB H W -> B BGR H W
x_hat_vggface = x_hat * 255 - self.vggface_caffe_RGB_mean
x_hat_vggface = x_hat_vggface[:,[2,1,0],:,:] # B RGB H W -> B BGR H W
```


---
### Explanations
The vgg19 and vggface loss mentioned in the paper are caffe trained version, the input should be in BGR order, [0-255].

However, in the original repo, vgg19 and vggface takes images in RGB order, and [0-1] normalized, while keeping the weights the same with paper, i.e.` vgg19_weight=1.5e-1, vggface_weight=2.5e-2`.

So either change the weight of the losses, or change the pretrained model to caffe pretrained version to balance the losses. 

For me, I download the caffe version of vgg19 from https://github.com/jcjohnson/pytorch-vgg, 
and make the input to vgg in range of [0-255], BGR order.

---

The following results are generated from the same person (id_08696) with different driving videos.

**Click the images to view video results on Youtube**

**1. Feed forward without finetuning**

[![](http://img.youtube.com/vi/HFI03fymvqI/0.jpg)](http://www.youtube.com/watch?v=HFI03fymvqI "")
[![](http://img.youtube.com/vi/Oz4AzhH5d0o/0.jpg)](http://www.youtube.com/watch?v=Oz4AzhH5d0o "")

**2. Fine tuning for 100 epochs**

[![](http://img.youtube.com/vi/WQ9z6GKu5_c/0.jpg)](http://www.youtube.com/watch?v=WQ9z6GKu5_c "")
[![](http://img.youtube.com/vi/O4WLB90m48U/0.jpg)](http://www.youtube.com/watch?v=O4WLB90m48U "")

As we can see, identity gap exists in feed forward results, but can be briged by finetuning. 

**3. More results:**

[![](http://img.youtube.com/vi/JBgnSG_t32M/0.jpg)](http://www.youtube.com/watch?v=JBgnSG_t32M "")
[![](http://img.youtube.com/vi/0gRR234skEA/0.jpg)](http://www.youtube.com/watch?v=0gRR234skEA "")

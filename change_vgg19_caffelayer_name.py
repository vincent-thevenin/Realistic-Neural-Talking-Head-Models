import torch
from torch.utils.model_zoo import load_url
from torchvision import models
from params.params import VGG19_caffe_weight_path

sd = torch.load(VGG19_caffe_weight_path)
sd['classifier.0.weight'] = sd['classifier.1.weight']
sd['classifier.0.bias'] = sd['classifier.1.bias']
del sd['classifier.1.weight']
del sd['classifier.1.bias']

sd['classifier.3.weight'] = sd['classifier.4.weight']
sd['classifier.3.bias'] = sd['classifier.4.bias']
del sd['classifier.4.weight']
del sd['classifier.4.bias']

torch.save(sd, VGG19_caffe_weight_path) 
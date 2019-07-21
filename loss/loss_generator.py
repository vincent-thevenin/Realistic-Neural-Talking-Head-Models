import torch
import torch.nn as nn
import imp
from torchvision.models import vgg19


class LossCnt(nn.Module):
    def __init__(self, VGGFace_body_path, VGGFace_weight_path, device):
        super(LossCnt, self).__init__()
        
        self.VGG19 = vgg19(pretrained=True)
        self.VGG19.eval()
        self.VGG19.to(device)
        
        MainModel = imp.load_source('MainModel', VGGFace_body_path)
        self.VGGFace = torch.load(VGGFace_weight_path, map_location = 'cpu')
        self.VGGFace.eval()
        self.VGGFace.to(device)

    def forward(self, x, x_hat, vgg19_weight=1e-2, vggface_weight=2e-3):
        l1_loss = nn.L1Loss()

        #define hook
        def vgg_x_hook(module, input, output):
            vgg_x_features.append(output.data)
        def vgg_xhat_hook(module, input, output):
            vgg_xhat_features.append(output.data)

        """same as above for vggface"""
        vgg_x_features = []
        vgg_xhat_features = []

        vgg_x_handles = []
        vgg_xhat_handles = []

        for i,m in enumerate(self.VGGFace.modules()):
            if type(m) is torch.nn.modules.Conv2d:
                vgg_x_handles.append(m.register_forward_hook(vgg_x_hook))

        self.VGGFace(x)
        for h in vgg_x_handles:
            h.remove()


        for i,m in enumerate(self.VGGFace.modules()):
            if type(m) is torch.nn.modules.Conv2d:
                vgg_xhat_handles.append(m.register_forward_hook(vgg_xhat_hook))

        self.VGGFace(x_hat)
        for h in vgg_xhat_handles:
            h.remove()

        lossface = 0
        shape_prev = vgg_x_features[0].shape
        #take everytime shape changes except first set of conv layers cf. paper
        for x_feat, xhat_feat in zip(vgg_x_features, vgg_xhat_features):
            if shape_prev != x_feat.shape:
                lossface += l1_loss(x_feat, xhat_feat)
                shape_prev = x_feat.shape

        """extract features for vgg19"""

        vgg_x_features = []
        vgg_xhat_features = []

        vgg_x_handles = []
        vgg_xhat_handles = []


        #place hooks
        for i,m in enumerate(self.VGG19.features.modules()):
            if type(m) is torch.nn.modules.Conv2d:
                vgg_x_handles.append(m.register_forward_hook(vgg_x_hook))

        #run model for x
        self.VGG19(x)

        #retrieve features
        for h in vgg_x_handles:
            h.remove()

        #place hooks
        for i,m in enumerate(self.VGG19.features.modules()):
            if type(m) is torch.nn.modules.Conv2d:
                vgg_xhat_handles.append(m.register_forward_hook(vgg_xhat_hook))

        #run for x_hat
        self.VGG19(x_hat)

        #retrieve features
        for h in vgg_xhat_handles:
            h.remove()

        loss19 = 0
        shape_prev = vgg_x_features[0].shape
        #take everytime shape changes except first set of conv layers cf. paper
        for x_feat, xhat_feat in zip(vgg_x_features, vgg_xhat_features):
            if shape_prev != x_feat.shape:
                loss19 += l1_loss(x_feat, xhat_feat)
                shape_prev = x_feat.shape




        loss = vgg19_weight * loss19 + vggface_weight * lossface

        return loss


class LossAdv(nn.Module):
    def __init__(self, FM_weight=1e1):
        super(LossAdv, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.FM_weight = FM_weight
        
    def forward(self, r_hat, D_res_list, D_hat_res_list):
        lossFM = 0
        for res, res_hat in zip(D_res_list, D_hat_res_list):
            lossFM += self.l1_loss(res, res_hat)
            
        return -r_hat.squeeze().mean() + lossFM * self.FM_weight


class LossMatch(nn.Module):
    def __init__(self, device, match_weight=8e1):
        super(LossMatch, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.match_weight = match_weight
        self.device = device
        
    def forward(self, e_vectors, W, i):
        loss = torch.zeros(e_vectors.shape[0],1).to(self.device)
        for b in range(e_vectors.shape[0]):
            for k in range(e_vectors.shape[1]):
                loss[b] += torch.abs(e_vectors[b,k].squeeze() - W[:,i[b]]).mean()
            loss[b] = loss[b]/e_vectors.shape[1]
        loss = loss.mean()
        return loss * self.match_weight
    
class LossG(nn.Module):
    """
	Loss for generator meta training
    Inputs: x, x_hat, r_hat, D_res_list, D_hat_res_list, e, W, i
    output: lossG
    """
    def __init__(self, VGGFace_body_path, VGGFace_weight_path, device, vgg19_weight=1e-2, vggface_weight=2e-3):
        super(LossG, self).__init__()
        
        self.LossCnt = LossCnt(VGGFace_body_path, VGGFace_weight_path, device)
        self.lossAdv = LossAdv()
        self.lossMatch = LossMatch(device=device)
        
    def forward(self, x, x_hat, r_hat, D_res_list, D_hat_res_list, e_vectors, W, i):
        loss_cnt = self.LossCnt(x, x_hat)
        loss_adv = self.lossAdv(r_hat, D_res_list, D_hat_res_list)
        loss_match = self.lossMatch(e_vectors, W, i)
        return loss_cnt + loss_adv + loss_match

class LossGF(nn.Module):
    """
	Loss for generator finetuning
    Inputs: x, x_hat, r_hat, D_res_list, D_hat_res_list, e, W, i
    output: lossG
    """
    def __init__(self, VGGFace_body_path, VGGFace_weight_path, device, vgg19_weight=1e-2, vggface_weight=2e-3):
        super(LossGF, self).__init__()
        
        self.LossCnt = LossCnt(VGGFace_body_path, VGGFace_weight_path, device)
        self.lossAdv = LossAdv()
        
    def forward(self, x, x_hat, r_hat, D_res_list, D_hat_res_list):
        loss_cnt = self.LossCnt(x, x_hat)
        loss_adv = self.lossAdv(r_hat, D_res_list, D_hat_res_list)
        return loss_cnt + loss_adv
import torch
import torch.nn.functional as F
import numpy as np

class HybridLoss(torch.nn.Module):
    def __init__(self, lamd=1e-1, spatial_tv=False, spectral_tv=False):
        super(HybridLoss, self).__init__()
        self.lamd = lamd
        self.use_spatial_TV = spatial_tv
        self.use_spectral_TV = spectral_tv
        self.fidelity = torch.nn.L1Loss()
        self.spatial = TVLoss(weight=1e-3)
        self.spectral = TVLossSpectral(weight=1e-3)

    def forward(self, y, gt):
        loss = self.fidelity(y, gt)
        spatial_TV = 0.0
        spectral_TV = 0.0
        if self.use_spatial_TV:
            spatial_TV = self.spatial(y)
        if self.use_spectral_TV:
            spectral_TV = self.spectral(y)
        total_loss = loss + spatial_TV + spectral_TV
        return total_loss


# from https://github.com/jxgu1016/Total_Variation_Loss.pytorch with slight modifications
class TVLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        # h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]).sum()
        # w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]).sum()
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class TVLossSpectral(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLossSpectral, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        c_x = x.size()[1]
        count_c = self._tensor_size(x[:, 1:, :, :])
        # c_tv = torch.abs((x[:, 1:, :, :] - x[:, :c_x - 1, :, :])).sum()
        c_tv = torch.pow((x[:, 1:, :, :] - x[:, :c_x - 1, :, :]), 2).sum()
        return self.TVLoss_weight * 2 * (c_tv / count_c) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class mse_sam_combo(torch.nn.Module):
    def __init__(self, N, lamd = 1e-1, mse_lamd=1, epoch=None):
        super(mse_sam_combo, self).__init__()
        self.N = N
        self.lamd = lamd
        self.mse_lamd = mse_lamd
        self.epoch = epoch
        return

    def forward(self, res, label):
        mse = F.mse_loss(res, label, reduction='mean') #resuction 'sum' is author propos
        # mse = func.l1_loss(res, label, size_average=False)
        loss = mse / (self.N * 2)
        esp = 1e-12
        H = label.size()[2]
        W = label.size()[3]
        Itrue = label.clone()
        Ifake = res.clone()
        nom = torch.mul(Itrue, Ifake).sum(dim=1)
        denominator = Itrue.norm(p=2, dim=1, keepdim=True).clamp(min=esp) * \
                      Ifake.norm(p=2, dim=1, keepdim=True).clamp(min=esp)
        denominator = denominator.squeeze()
        # sam = -np.pi/2*torch.div(nom, denominator) + np.pi/2
        sam = torch.div(nom, denominator).acos()
        sam[sam!=sam] = 0  # Handle NaN values
        sam_sum = torch.sum(sam) / (self.N * H * W)
        if self.epoch is None:
            total_loss = self.mse_lamd * loss + self.lamd * sam_sum
        else:
            norm = self.mse_lamd + self.lamd * 0.1 **(self.epoch//10)
            lamd_sam = self.lamd * 0.1 ** (self.epoch // 10)
            total_loss = self.mse_lamd/norm * loss + lamd_sam/norm * sam_sum
        return total_loss
    
class L1_SAM_Loss(torch.nn.Module): #my little
    def __init__(self):
        super(L1_SAM_Loss, self).__init__()
        self.l1_loss = torch.nn.L1Loss()
        
    def forward(self, target, reference):
        L1 = self.l1_loss(target, reference)

        SAM = sum()

class OnlySAMLoss(torch.nn.Module):
    def __init__(self, eps=1e-12):
        super(OnlySAMLoss, self).__init__()
        self.eps = eps

    def forward(self, res, label):
        H, W = label.size()[2], label.size()[3]
        nom = torch.sum(res * label, dim=1)
        denom = (
            res.norm(p=2, dim=1) * label.norm(p=2, dim=1)
        ).clamp(min=self.eps)
        cos_theta = nom / denom
        cos_theta = torch.clamp(cos_theta, -1 + self.eps, 1 - self.eps)
        sam = torch.acos(cos_theta)
        return torch.mean(sam)  # averaged SAM loss over batch



##FROM CST: 
class HLoss(torch.nn.Module):
    def __init__(self, la1, la2, sam=True, gra=True):
        super(HLoss, self).__init__()
        self.lamd1 = la1
        self.lamd2 = la2
        self.sam = sam
        self.gra = gra

        self.fidelity = torch.nn.L1Loss()
        self.gra = torch.nn.L1Loss()

    def forward(self, y, gt):
        loss1 = self.fidelity(y, gt)
        loss2 = self.lamd1 * cal_sam(y, gt)
        loss3 = self.lamd2 * self.gra(cal_gradient(y), cal_gradient(gt))
        loss = loss1 + loss2 + loss3
        return loss

def cal_sam(Itrue, Ifake):
  esp = 1e-6
  InnerPro = torch.sum(Itrue*Ifake,1,keepdim=True)
  len1 = torch.norm(Itrue, p=2,dim=1,keepdim=True)
  len2 = torch.norm(Ifake, p=2,dim=1,keepdim=True)
  divisor = len1*len2
  mask = torch.eq(divisor,0)
  divisor = divisor + (mask.float())*esp
#   cosA = torch.sum(InnerPro/divisor,1).clamp(-1+esp, 1-esp)
  cosA = torch.sum(InnerPro/divisor,1).clamp(-0.9999, 0.9999)
  sam = torch.acos(cosA)
  sam = torch.mean(sam) / np.pi
  return sam

def cal_gradient_c(x):
    c_x = x.size(1)
    g = x[:, 1:, 1:, 1:] - x[:, :c_x - 1, 1:, 1:]
    return g

def cal_gradient_x(x):
    c_x = x.size(2)
    g = x[:, 1:, 1:, 1:] - x[:, 1:, :c_x - 1, 1:]
    return g

def cal_gradient_y(x):
    c_x = x.size(3)
    g = x[:, 1:, 1:, 1:] - x[:, 1:, 1:, :c_x - 1]
    return g


def cal_gradient(inp):
    x = cal_gradient_x(inp)
    y = cal_gradient_y(inp)
    c = cal_gradient_c(inp)
    g = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2) + torch.pow(c, 2) + 1e-6)
    return g

class SAM_L1_Loss(torch.nn.Module):
    def __init__(self):
        super(SAM_L1_Loss, self).__init__()
        self.h_loss = HLoss(0.3,0.1)

    def forward(self, pred, label):
        # --- L1 Loss ---
        l1_loss = F.l1_loss(pred, label, reduction='mean')

        # SAM Loss 
        eps = 1e-8
        B, C, H, W = label.shape
        pred_flat = pred.view(B, C, -1)
        label_flat = label.view(B, C, -1)

        dot = torch.sum(pred_flat * label_flat, dim=1)
        norm_pred = torch.norm(pred_flat, dim=1)
        norm_label = torch.norm(label_flat, dim=1)
        cos_theta = dot / (norm_pred * norm_label + eps)
        # cos_theta = torch.clamp(cos_theta, -1.0, 1.0) #this gave nan loss error in washinton dc mall
        cos_theta = torch.clamp(cos_theta, -0.9999, 0.9999)
        sam_loss = torch.acos(cos_theta).mean()
        fft_loss = FFTLoss()(pred, label)
        hybrid = HybridLoss(lamd=1e-3, spatial_tv=True, spectral_tv=True)(pred, label)

        h_loss = self.h_loss(pred, label)

        # Total Loss
        total_loss = l1_loss + 0.1 * sam_loss + fft_loss #hybrid #l1_loss# + fft_loss # + 0.1 * sam_loss + h_loss  #+ hybrid #
        return total_loss


#https://github.com/sunny2109/SAFMN/blob/00710f1fccf6b5aa7d40f2b474b4d14689b716a8/basicsr/losses/losses.py#L258
class FFTLoss(torch.nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)
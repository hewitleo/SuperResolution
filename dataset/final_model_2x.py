import torch
import torchvision
import kornia
from kornia.filters import laplacian
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

def AddWeighted(img1, alpha, img2, beta, lambda_=0.0):
    img = (img1 * torch.multiply(torch.ones(img1.shape, dtype=torch.uint8, device=img1.device), alpha)) + (img2 * torch.multiply(torch.ones(img2.shape, dtype=torch.uint8, device=img1.device), beta)) + lambda_
    return img.round().float()

def laplacian_kernels():
    kernel = torch.tensor([[2., 0., 2.], [0., -8., 0.], [2., 0., 2.]], dtype=torch.float32)[None, None, :, :]
    return kernel

def sobel_kernels():
    kernel_x = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32)[None, None, :, :]
    kernel_y = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32)[None, None, :, :]
    return kernel_x, kernel_y

def prewitt_kernels():
    kernel_x = torch.tensor([[1., 1., 1.], [0., 0., 0.], [-1., -1., -1.]], dtype=torch.float32)[None, None, :, :]
    kernel_y = torch.tensor([[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]], dtype=torch.float32)[None, None, :, :]
    return kernel_x, kernel_y

def roberts_kernels():
    kernel_x = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)[None, None, :, :]
    kernel_y = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float32)[None, None, :, :]
    return kernel_x, kernel_y

class Block1(nn.Module):
    def __init__(self, channels = 128):
        super(Block1, self).__init__()

        self. channels = channels

        self.gaussian_filter = nn.Conv2d(channels, channels, 3, padding='same', bias=False, padding_mode='reflect', groups=channels)

        self.sobel_x = nn.Conv2d(channels, channels, 3, bias=False, padding='same', padding_mode='reflect', groups=channels)
        self.sobel_y = nn.Conv2d(channels, channels, 3, bias=False, padding='same', padding_mode='reflect', groups=channels)

        self.prewitt_x = nn.Conv2d(channels, channels, 3, bias=False, padding='same', padding_mode='reflect', groups=channels)
        self.prewitt_y = nn.Conv2d(channels, channels, 3, bias=False, padding='same', padding_mode='reflect', groups=channels)

        self.roberts_x = nn.Conv2d(channels, channels, 3, bias=False, padding='same', padding_mode='reflect', groups=channels)
        self.roberts_y = nn.Conv2d(channels, channels, 3, bias=False, padding='same', padding_mode='reflect', groups=channels)

        self.laplacian = nn.Conv2d(channels, channels, 3, bias=False, padding='same', padding_mode='reflect', groups=channels)

        # self.sobel_x.weight.data, self.sobel_y.weight.data = sobel_kernels()
        # self.prewitt_x.weight.data, self.prewitt_y.weight.data = prewitt_kernels()
        # self.roberts_x.weight.data, self.roberts_y.weight.data = roberts_kernels()
        # self.laplacian.weight.data = laplacian_kernels()

        sobel_x_kernel, sobel_y_kernel = sobel_kernels()
        self.sobel_x.weight.data = sobel_x_kernel.repeat(channels, 1, 1, 1)
        self.sobel_y.weight.data = sobel_y_kernel.repeat(channels, 1, 1, 1)

        prewitt_x_kernel, prewitt_y_kernel = prewitt_kernels()
        self.prewitt_x.weight.data = prewitt_x_kernel.repeat(channels, 1, 1, 1)
        self.prewitt_y.weight.data = prewitt_y_kernel.repeat(channels, 1, 1, 1)

        roberts_x_kernel, roberts_y_kernel = roberts_kernels()
        self.roberts_x.weight.data = roberts_x_kernel.repeat(channels, 1, 1, 1)
        self.roberts_y.weight.data = roberts_y_kernel.repeat(channels, 1, 1, 1)

        laplacian_kernel = laplacian_kernels()
        self.laplacian.weight.data = laplacian_kernel.repeat(channels, 1, 1, 1)


        # self.gaussian_filter.weight.data = torch.tensor([[0.0625, 0.125, 0.0625],
        #                                                  [0.125, 0.25, 0.125],
        #                                                  [0.0625, 0.125, 0.0625]],
        #                                                 dtype=torch.float32)[None, None, :, :]
        gaussian_kernel = torch.tensor([[0.0625, 0.125, 0.0625],
                                        [0.125,  0.25,  0.125],
                                        [0.0625, 0.125, 0.0625]], dtype=torch.float32)
    
        gaussian_kernel = gaussian_kernel[None, None, :, :].repeat(channels, 1, 1, 1)
        self.gaussian_filter.weight.data = gaussian_kernel


        self.gaussian_filter.weight.requires_grad = False
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False

        self.prewitt_x.weight.requires_grad = False
        self.prewitt_y.weight.requires_grad = False

        self.roberts_x.weight.requires_grad = False
        self.roberts_y.weight.requires_grad = False

        self.laplacian.weight.requires_grad = False

        self.conv_1by1 = torch.nn.Conv2d(channels * 6, channels, kernel_size=1, padding=0)

    def forward(self, HSI_IP):                  

            # pan_lp = filters.gaussian_blur2d(pan, (3, 3), (self.sigma, self.sigma), separable=False)
            pan_lp = self.gaussian_filter(HSI_IP)

            norm = torch.amax(pan_lp, dim=(1, 2, 3), keepdim=True)

            pan_lp = pan_lp * 255.0 / norm

            sobel_x = torch.clip(torch.abs(self.sobel_x(pan_lp)), 0, 255).round()
            sobel_y = torch.clip(torch.abs(self.sobel_y(pan_lp)), 0, 255).round()
            sobel = AddWeighted(sobel_x.int(), 0.5, sobel_y.int(), 0.5, 0.0)
            sobel = sobel * norm / 255.0

            pan_lp = pan_lp // 1.0

            prewitt_x = torch.clip(torch.abs(self.prewitt_x(pan_lp)), 0, 255).round()
            prewitt_y = torch.clip(torch.abs(self.prewitt_y(pan_lp)), 0, 255).round()
            prewitt = AddWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0.0)
            prewitt = prewitt * norm / 255.0

            roberts_x = self.roberts_x(pan_lp)
            roberts_y = self.roberts_y(pan_lp)

            roberts_x = torch.clip(torch.abs(roberts_x), 0, 255).round()
            roberts_y = torch.clip(torch.abs(roberts_y), 0, 255).round()
            roberts = AddWeighted(roberts_x, 0.5, roberts_y, 0.5, 0.0)
            roberts = roberts * norm / 255.0

            laplacian = self.laplacian(pan_lp)
            laplacian = torch.clip(torch.abs(laplacian), 0, 255).round()
            laplacian = laplacian * norm / 255.0

            pan_pad_x = F.pad(HSI_IP, (1, 0, 0, 0), mode='replicate')
            pan_pad_y = F.pad(HSI_IP, (0, 0, 1, 0), mode='replicate')

            diff_x = - pan_pad_x[:, :, :, :-1] + pan_pad_x[:, :, :, 1:]
            diff_y = - pan_pad_y[:, :, :-1, :] + pan_pad_y[:, :, 1:, :]

            x = torch.cat([diff_y, diff_x, roberts, prewitt, sobel, laplacian], dim=1)

            x = self.conv_1by1(x)

            return x

def variance_scaling_initializer(tensor):
    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):  ## initialization for Conv2d
                variance_scaling_initializer(m.weight)  # method 1: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):  ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):  ## initialization for nn.Linear
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

class MultiScaleExtractor(nn.Module): #my code
    def __init__(self):
        super(MultiScaleExtractor, self).__init__() #4,8,10,12,14 before used
        self.conv1 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=6, dilation=6)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=12, dilation=12)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=18, dilation=18)
        self.GlobalPool = nn.AdaptiveAvgPool2d((1,1))
        self.conv10 = nn.Conv2d(128 * 5, 128, kernel_size=1, padding=0)

    def forward(self,x):
        input = x
        x1 = self.conv1(input)
        x2 = self.conv2(input)
        x6 = self.conv6(input)
        x8 = self.conv8(input)
        x10 = self.GlobalPool(input)
        x10 = F.interpolate(x10, size = (x.shape[2], x.shape[2]), mode = 'bilinear', align_corners=False)
        x11 = torch.cat([x1,x2,x6,x8,x10], dim=1)
        x12 = self.conv10(x11)
        return x12


class Block2(nn.Module): # has channel and spatial attention
    def __init__(self, channels, n_features=128):
        super(Block2, self).__init__()
        self.conv1 = nn.Conv2d(channels, n_features, 3, bias=True, padding='same')
        self.conv2 = nn.Conv2d(channels, n_features, 5, bias=True, padding='same')
        self.conv3 = nn.Conv2d(channels, n_features, 7, bias=True, padding='same')
        self.conv4 = nn.Conv2d(128 * 3, 128, 1, bias=True, padding=0)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x4 = torch.cat([x1, x2, x3], dim=1)
        x5 = F.relu(self.conv4(x4))
        return x5#output
    
class Block2_grp(nn.Module): # has channel and spatial attention
    def __init__(self, channels, n_features=128):
        super(Block2_grp, self).__init__()
        self.conv1 = nn.Conv2d(channels, n_features, 3, bias=True, padding='same', groups=128)
        self.conv2 = nn.Conv2d(channels, n_features, 5, bias=True, padding='same', groups = 128)
        self.conv3 = nn.Conv2d(channels, n_features, 7, bias=True, padding='same', groups= 128)
        self.conv4 = nn.Conv2d(128 * 3, 128, 1, bias=True, padding=0)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x4 = torch.cat([x1, x2, x3], dim=1)
        x5 = F.relu(self.conv4(x4))
        return x5#output

class InceptionBlock(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(InceptionBlock, self).__init__()

        branch_channels = out_channels // 4

        # 1x1 Convolution
        self.branch1 = nn.Conv2d(in_channels, branch_channels, kernel_size=1)

        # 3x3 Convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # 5x5 Convolution
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # 3x3 MaxPool followed by 1x1 Conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1)
        )

        # Final output projection
        self.project = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)

        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.project(out)
        return F.relu(out)

class ResBlock_v1(torch.nn.Module): #3 by 3 normal
    def __init__(self, in_channels, out_channels):
        super(ResBlock_v1, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x):
        # res = x
        res = self.skip_proj(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + res
        x = F.relu(x, inplace=True)
        return x

class ResBlock_v2(torch.nn.Module): #3 by 3 dilation 2
    def __init__(self, in_channels, out_channels):
        super(ResBlock_v2, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x):
        # res = x
        res = self.skip_proj(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + res
        x = F.relu(x, inplace=True)
        return x
    
class ResBlock_v3(torch.nn.Module): #3 by 3 dilation 3
    def __init__(self, in_channels, out_channels):
        super(ResBlock_v3, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x):
        # res = x
        res = self.skip_proj(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + res
        x = F.relu(x, inplace=True)
        return x

class ResBlock_v4(torch.nn.Module): # 5 by 5 kernel
    def __init__(self, in_channels, out_channels):
        super(ResBlock_v4, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)#, dilation=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)#, dilation=3, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x):
        # res = x
        res = self.skip_proj(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + res
        x = F.relu(x, inplace=True)
        return x

class ResBlock_v5(torch.nn.Module): # 7 by 7 kernel
    def __init__(self, in_channels, out_channels):
        super(ResBlock_v5, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)#, dilation=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3)#, dilation=3, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x):
        # res = x
        res = self.skip_proj(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + res
        x = F.relu(x, inplace=True)
        return x
    
class ResChain_v1(nn.Module): #3 normal res connections and again long skip
    def __init__(self, in_channels, out_channels):
        super(ResChain_v1, self).__init__()
        self.res_1 = ResBlock_v1(in_channels, out_channels)
        self.res_2 = ResBlock_v1(out_channels, out_channels)
        self.res_3 = ResBlock_v1(out_channels, out_channels)
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x):
        # res = x
        res = self.skip_proj(x)
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = x + res
        return x

class ResChain_v2(nn.Module): #3 res connections DIALTION 2 AGAIN and again long skip
    def __init__(self, in_channels, out_channels):
        super(ResChain_v2, self).__init__()
        self.res_1 = ResBlock_v2(in_channels, out_channels)
        self.res_2 = ResBlock_v2(out_channels, out_channels)
        self.res_3 = ResBlock_v2(out_channels, out_channels)
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x):
        # res = x
        res = self.skip_proj(x)
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = x + res
        return x

class ResChain_v3(nn.Module): #3 res connections DIALATION 3 AGAIN and again long skip
    def __init__(self, in_channels, out_channels):
        super(ResChain_v3, self).__init__()
        self.res_1 = ResBlock_v3(in_channels, out_channels)
        self.res_2 = ResBlock_v3(out_channels, out_channels)
        self.res_3 = ResBlock_v3(out_channels, out_channels)
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x):
        # res = x
        res = self.skip_proj(x)
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = x + res
        return x

class ResChain_v4(nn.Module): #3 res connections 7, 5, 3 and again long skip
    def __init__(self, in_channels, out_channels):
        super(ResChain_v4, self).__init__()
        self.res_1 = ResBlock_v5(in_channels, out_channels)
        self.res_2 = ResBlock_v4(out_channels, out_channels)
        self.res_3 = ResBlock_v1(out_channels, out_channels)
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x):
        # res = x
        res = self.skip_proj(x)
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = x + res
        return x

# class ResDeformalBlock(nn.Module): # 1 deform , 1 cnn 3 by 3
#     def __init__(self, input_channels):
#         super(ResDeformalBlock, self).__init__()
#         self.offset = nn.Conv2d(input_channels, 18, 3, padding=1)
#         self.deform_conv = torchvision.ops.DeformConv2d(input_channels, 128, 3, padding=1)
#         self.conv = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.res_block = ResBlock_v1(128, 128)
#     def forward(self, x):
#         offset = self.offset(x)
#         x = self.deform_conv(x, offset)
#         x = self.conv(x)
#         x = self.res_block(x)
#         return x

class ResDeformBlock_v1(nn.Module):  # a deformable residual block
    def __init__(self, input_channels):
        super().__init__()
        self.offset = nn.Conv2d(input_channels, 18, 3, padding=1)
        self.deform_conv = torchvision.ops.DeformConv2d(input_channels, 128, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.offset2 = nn.Conv2d(128, 18, 3, padding=1)
        self.deform_conv2 = torchvision.ops.DeformConv2d(128, 128, 3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = x
        offset1 = self.offset(x)
        x = self.deform_conv(x, offset1)
        x = self.relu1(x)
        offset2 = self.offset2(x)
        x = self.deform_conv2(x, offset2)
        x = self.relu2(x)
        return x + residual

class ResDeformChain_v1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res_1 = ResDeformBlock_v1(in_channels)
        self.res_2 = ResDeformBlock_v1(in_channels)
        self.res_3 = ResDeformBlock_v1(in_channels)

    def forward(self, x):
        res = x
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        return x + res


class Upsample_pixel_shuffle(torch.nn.Module): # 1 conv ops 128 * 4, ps .
    def __init__(self, kernel_size, padding, in_channels):
        super(Upsample_pixel_shuffle, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 128*4, kernel_size=kernel_size, padding=padding)
        self.pixel_shuffle = torch.nn.PixelShuffle(2)
        # self.conv2 = torch.nn.Conv2d(128, 128*4, kernel_size=3, padding=1)
        # self.pixel_shuffle_1 = torch.nn.PixelShuffle(2)

    def forward(self, x):   
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(x)
        # x = F.relu(self.conv2(x))
        # x = self.pixel_shuffle(x)
        return x

class Downsample_pixel_unshuffle(nn.Module):
    def __init__(self):
        super(Downsample_pixel_unshuffle, self).__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.conv = nn.Conv2d(128 * 4, 128, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.pixel_unshuffle(x)
        x = F.relu(self.conv(x))
        return x

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=True):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
        init_weights(self.conv_du, self.avg_pool)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class FeatureExploitor(nn.Module): # has 4 conv layers
    def __init__(self, input_channels):
        super(FeatureExploitor, self).__init__()
        self.conv_1 = torch.nn.Conv2d(input_channels, 128, kernel_size=7, padding=3, groups=128)
        self.conv_2 = torch.nn.Conv2d(128, 128, kernel_size=5, padding=2, groups=128)
        # self.conv_3 = torch.nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128)
        # self.conv_4 = torch.nn.Conv2d(128, 128, kernel_size=5, padding=2, groups=128)
        self.conv_5 = torch.nn.Conv2d(128 * 2, 128, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = F.relu(self.conv_1(x))
        x2 = F.relu(self.conv_2(x))
        # x3 = F.relu(self.conv_3(x))
        # x4 = F.relu(self.conv_4(x))
        x5 = self.conv_5(torch.cat([x1,x2], dim=1))
        return x5

class DeformalBlock(nn.Module): # 1 deform , 1 cnn 3 by 3
    def __init__(self, input_channels):
        super(DeformalBlock, self).__init__()
        self.offset = nn.Conv2d(input_channels, 18, 3, padding = 1)
        self.deform_conv = torchvision.ops.DeformConv2d(input_channels, 128, 3, padding = 1)
        self.conv = torch.nn.Conv2d(128,128, kernel_size=3, padding=1)

    def forward(self, x):
        offset = self.offset(x)
        x = self.deform_conv(x, offset)
        #x = self.conv(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, scale, channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels * (scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

class SimpleMDSR(nn.Module):
    def __init__(self, num_blocks=8, num_feats=64, in_channels=128, out_channels=128):
        super(SimpleMDSR, self).__init__()
        self.head = nn.Conv2d(in_channels, num_feats, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[SimpleResBlock(num_feats) for _ in range(num_blocks)])
        self.upsample = UpsampleBlock(scale=2, channels=num_feats)
        self.tail = nn.Conv2d(num_feats, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        up = self.upsample(res)
        out = self.tail(up)
        return out

class FSRCNN_PixelShuffle(nn.Module):
    def __init__(self, in_channels=128, scale=2):
        super(FSRCNN_PixelShuffle, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1024, 5, 1, 2)
        self.prelu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(1024, 256, 1)
        self.prelu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.prelu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.prelu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(256, 1024, 1)
        self.prelu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(1024, in_channels * (scale ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):

        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        skip = x
        x = self.prelu3(self.conv3(x))
        x = x + skip
        x = self.prelu4(self.conv4(x))
        x = x + skip
        x = self.prelu5(self.conv5(x))
        x = self.conv6(x)
        x = self.pixel_shuffle(x)
        return x

class FSRCNN_Block(nn.Module):
    def __init__(self, in_channels, scale=2):
        super(FSRCNN_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, 5, 1, 2)
        self.prelu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 64, 1)
        self.prelu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.prelu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.prelu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(64, 128, 1)
        self.prelu5 = nn.ReLU()

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        skip = x
        x = self.prelu3(self.conv3(x))
        x = x + skip
        x = self.prelu4(self.conv4(x))
        x = x + skip
        x = self.prelu5(self.conv5(x))
        return x

class FSRCNN_PixelShuffle_SSPSRStyle(nn.Module):
    def __init__(self, in_channels=128, scale=2):
        super(FSRCNN_PixelShuffle_SSPSRStyle, self).__init__()
        self.group_channels = in_channels // 3
        self.block1 = FSRCNN_Block(self.group_channels, scale)
        self.block2 = FSRCNN_Block(self.group_channels, scale)
        self.block3 = FSRCNN_Block(in_channels - 2 * self.group_channels, scale)

        self.block_1 = Block1(128)
        self.block_2 = Block2(128)

        self.fuse = nn.Conv2d(5 * 128, in_channels * (scale ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):

        x_up = F.interpolate(x, size=(x.shape[2]*2, x.shape[2]*2), mode='bilinear', align_corners=False)

        g1 = x[:, :self.group_channels, :, :]
        g2 = x[:, self.group_channels:2*self.group_channels, :, :]
        g3 = x[:, 2*self.group_channels:, :, :]

        f1 = self.block1(g1)
        f2 = self.block2(g2)
        f3 = self.block3(g3)

        a1 = self.block_1(x)
        c1 = self.block_2(x)
        

        fused = torch.cat([f1, f2, f3, a1, c1], dim=1)
        out = self.fuse(fused)
        out = self.pixel_shuffle(out)
        return out

class FSRCNN_Block(nn.Module):
    def __init__(self, in_channels, scale=2):
        super(FSRCNN_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, 5, 1, 2)
        self.prelu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 64, 1)
        self.prelu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.prelu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.prelu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(64, 128, 1)
        self.prelu5 = nn.ReLU()
        self.final = nn.Conv2d(128, in_channels * (scale ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        skip = x
        x = self.prelu3(self.conv3(x))
        x = x + skip
        x = self.prelu4(self.conv4(x))
        x = x + skip
        x = self.prelu5(self.conv5(x))
        x = self.final(x)
        x = self.pixel_shuffle(x)
        return x

class FSRCNN_PixelShuffle_SSPSRStyle(nn.Module):
    def __init__(self, in_channels=128, scale=2):
        super(FSRCNN_PixelShuffle_SSPSRStyle, self).__init__()
        self.group_channels = in_channels // 3
        remaining = in_channels - 2 * self.group_channels

        self.block1 = FSRCNN_Block(self.group_channels, scale)
        self.block2 = FSRCNN_Block(self.group_channels, scale)
        self.block3 = FSRCNN_Block(remaining, scale)
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)

    def forward(self, x):
        g1 = x[:, :self.group_channels, :, :]
        g2 = x[:, self.group_channels:2*self.group_channels, :, :]
        g3 = x[:, 2*self.group_channels:, :, :]

        f1 = self.block1(g1)  # [B, C, H*scale, W*scale]
        f2 = self.block2(g2)
        f3 = self.block3(g3)

        out = torch.cat([f1, f2, f3], dim=1)  # concat in channel dim
        out = self.conv1(out)
        return out

class Conv20Block(nn.Module):
    def __init__(self, channels=128):
        super(Conv20Block, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Add 20 convolutional layers with varying kernel sizes
        for i in range(50):
            if i % 3 == 0:
                kernel_size = 1
                padding = 0
            elif i % 3 == 1:
                kernel_size = 3
                padding = 1
            else:
                kernel_size = 9
                padding = 4

            conv = nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding
            )
            self.layers.append(conv)

        # Final 1x1 convolution to fuse features
        self.final_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        out = x
        for conv in self.layers:
            out = F.relu(conv(out))
        out = self.final_conv(out)
        return out

class GDRRN(nn.Module):
    def __init__(self, input_channels=128, num_features=128, groups=4, num_recursions=6):
        super(GDRRN, self).__init__()

        # Initial feature extraction (not grouped)
        self.input = nn.Conv2d(input_channels, num_features, kernel_size=3, padding=1, bias=False)

        # Grouped convolutional layers inside recursive block
        self.group_conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, groups=groups, bias=False)
        self.group_conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, groups=groups, bias=False)

        # Final output layer to map back to original channels (not grouped)
        self.output = nn.Conv2d(num_features, input_channels, kernel_size=3, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.num_recursions = num_recursions

    def forward(self, x):
        # Save residual for final skip connection
        residual = x  # x is already high-resolution

        # Initial feature extraction
        features = self.input(self.relu(x))
        out = features

        # Recursive grouped conv blocks
        for _ in range(self.num_recursions):
            out = self.group_conv1(self.relu(out))
            out = self.group_conv2(self.relu(out))
            out = out + features  # local skip connection

        # Final reconstruction
        out = self.output(self.relu(out))

        # Global residual connection
        out = out + residual

        return out

class HSISR_new(torch.nn.Module):
    def __init__(self):
        super(HSISR_new, self).__init__()

        self.deformConv_1 = DeformalBlock(128)
        self.deformConv_2 = DeformalBlock(128)
        self.deformConv_3 = DeformalBlock(128)
        self.block_1 = Block1(128)
        # self.block_2 = Block2(128, 128)

        self.multi_scale_1 = MultiScaleExtractor()
        self.multi_scale_2 = MultiScaleExtractor()
        self.multi_scale_3 = MultiScaleExtractor()

        self.chnlatt_1 = CALayer(128)
        self.chnlatt_2 = CALayer(128)
        self.chnlatt_3 = CALayer(128)
        self.chnlatt_4 = CALayer(128)
        # self.gdrnn = GDRRN()

        # self.block_2_grp = Block2_grp(128, 128)

        self.conv_1 = torch.nn.Conv2d(128 * 3, 256, kernel_size=1, padding=0, stride=1)
        self.conv_2 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        #temp trial
        self.conv_3 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.conv_4 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.conv_5 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.conv_6 = torch.nn.Conv2d(512, 128, kernel_size=3, padding=1, stride=1)
        self.conv_7 = torch.nn.Conv2d(128 *3, 128, kernel_size=3, padding=1, stride=1)
        self.conv_8 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.conv_9 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.conv_10 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)

        self.res_1 = ResBlock_v1(128*3, 128*3)
        self.res_2 = ResBlock_v4(128, 128)
        self.res_3 = ResBlock_v5(128, 128)
        self.res_4 = ResBlock_v1(128, 128)
        # self.res_5 = ResBlock_v1(128, 128)

        # self.deep_res_chain_1 = nn.Sequential(*[ResBlock_v1(128, 128) for _ in range(10)])
        # self.deep_res_chain_2 = nn.Sequential(*[ResBlock_v1(128) for _ in range(10)])
        # self.deep_res_chain_3 = nn.Sequential(*[ResBlock_v1(128) for _ in range(10)])
        # self.deep_res_chain_4 = nn.Sequential(*[ResBlock_v1(128) for _ in range(40)])

        self.res_chain_1 = ResChain_v1(256, 256)
        self.res_chain_2 = ResChain_v2(256, 256)
        self.res_chain_3 = ResChain_v3(256, 256)

        self.res_chain_4 = ResChain_v1(256, 256)
        self.res_chain_5 = ResChain_v2(256, 256)
        self.res_chain_6 = ResChain_v3(256, 256)

        self.res_chain_7 = ResChain_v1(128*3, 128)
        self.res_chain_8 = ResChain_v2(256, 512)
        self.res_chain_9 = ResChain_v3(256, 512)
        self.res_chain_v4_10 = ResChain_v4(128, 256) # kerels 7 5 3

        self.inception_block_1 = InceptionBlock(in_channels=128, out_channels=128)

        # self.deform_res_chain_1 = ResDeformChain_v1(128, 128)
        # self.deform_res_chain_2 = ResDeformChain_v1(128, 128)

        self.pix_shuff_1 = Upsample_pixel_shuffle(kernel_size=3, padding=1, in_channels=256)
        self.pix_shuff_2 = Upsample_pixel_shuffle(kernel_size=3, padding=1, in_channels=256)
        self.pix_shuff_3 = Upsample_pixel_shuffle(kernel_size=3, padding=1, in_channels=256)
        self.pix_shuff_4 = Upsample_pixel_shuffle(kernel_size=3, padding=1, in_channels=128)
        # self.pix_unshuff_1 = Downsample_pixel_unshuffle()
        # self.bn = nn.BatchNorm2d(128)
        # self.bn_1 = nn.BatchNorm2d(384)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        input = x
        glob_skip = F.interpolate(x, size=(x.shape[2]*2, x.shape[2]*2), mode='bilinear', align_corners=False)
        # ps_us = self.pix_unshuff_1(glob_skip)
        skip = self.conv_3(x)
        pix_skip = self.pix_shuff_4(input)

        res_2 = self.res_2(input)
        res_3 = self.res_3(input)
        res_4 = self.res_4(input)

        concat_3_blks = torch.cat([res_3, res_2, res_4], dim=1)
        # concat_3_blks = self.maxpool_1(concat_3_blks)
        concat_3_blks = self.conv_1(concat_3_blks)

        res_chain_4 = self.res_chain_1(concat_3_blks)
        res_chain_5 = self.res_chain_2(concat_3_blks)
        res_chain_6 = self.res_chain_3(concat_3_blks)

        ps_1 = self.pix_shuff_1(res_chain_4)
        ps_2 = self.pix_shuff_2(res_chain_5)
        ps_3 = self.pix_shuff_3(res_chain_6)

        concat = torch.cat([ps_1, ps_2, ps_3], dim= 1)
        concat = self.res_1(concat) ############very powerful
        concat = self.conv_7(concat)
        # concat = self.res_chain_7(concat)
        concat = self.chnlatt_4(concat)
        concat = concat + pix_skip
        # concat = concat + self.conv_4(glob_skip)
        concat = self.conv_3(concat)
        return concat
    



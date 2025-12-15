import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock_v1(torch.nn.Module): #3 by 3 normal
    def __init__(self, in_channels, out_channels, kernel_size, dilation = 1, groups=1):
        super(ResBlock_v1, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding= dilation * (kernel_size-1)//2, groups=groups, dilation=dilation, bias=False)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding= dilation * (kernel_size-1)//2, groups=groups, dilation=dilation, bias=False)
        self.act = nn.PReLU() 
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x):
        res = self.skip_proj(x)

        x = self.conv1(x)
        # x = self.act(x)
        x = F.relu(x, inplace=True)
        # x = F.silu(x)
        x = self.conv2(x)

        x = x + res
        # x = F.relu(x, inplace=True)
        return x


class DenseResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DenseResBlock, self).__init__()

        self.res_1 = ResBlock_v1(in_channels, out_channels, kernel_size, groups=1)
        # self.mix_1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        # self.ca_1 = CALayer(out_channels)

        self.res_2 = ResBlock_v1(in_channels + out_channels, out_channels, kernel_size, groups=1)
        # self.mix_2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        # self.ca_2 = CALayer(out_channels)

        self.res_3 = ResBlock_v1(in_channels + 2 * out_channels, out_channels, kernel_size, groups=1)
        # self.mix_3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        # self.ca_3 = CALayer(out_channels)

        self.res_4 = ResBlock_v1(in_channels + 3 * out_channels, out_channels, kernel_size, groups=1)
        # self.mix_4 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        # self.ca_4 = CALayer(out_channels)

        self.final_proj = nn.Conv2d(in_channels + 4 * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out1 = self.res_1(x)
        # out1 = self.mix_1(out1)
        # out1 = self.ca_1(out1)
        out2 = self.res_2(torch.cat([x, out1], dim=1))
        # out2 = self.mix_2(out2)
        # out2 = self.ca_2(out2)
        out3 = self.res_3(torch.cat([x, out1, out2], dim=1))
        # out3 = self.mix_3(out3)
        # out3 = self.ca_3(out3)
        out4 = self.res_4(torch.cat([x, out1, out2, out3], dim=1))
        # out4 = self.mix_4(out4)
        # out4 = self.ca_4(out4)
 
        # out5 = self.res_5(torch.cat([x, out1, out2, out3, out4], dim=1))
        # out6 = self.res_6(torch.cat([x, out1, out2, out3, out4, out5], dim=1))
  
        concat_all = torch.cat([x, out1, out2, out3, out4],dim=1)#, out5, out6, out7, out8, out9], dim=1)
        out = self.final_proj(concat_all)
        return out


class Upsample_pixel_shuffle(torch.nn.Module): # 1 conv ops 128 * 4, ps .
    def __init__(self, kernel_size, padding, in_channels):
        super(Upsample_pixel_shuffle, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels*4, kernel_size=kernel_size, padding=padding)
        self.pixel_shuffle = torch.nn.PixelShuffle(2)
        # self.conv2 = torch.nn.Conv2d(256, 256*4, kernel_size=3, padding=1)
        # self.pixel_shuffle_1 = torch.nn.PixelShuffle(2)

    def forward(self, x):   
        # x = F.relu(self.conv1(x))
        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        # x = F.relu(self.conv2(x))
        # x = self.pixel_shuffle(x)
        return x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=True):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## From SSPSR: https://github.com/junjun-jiang/SSPSR/blob/master/common.py
class ResBlock_main(nn.Module):
    def __init__(self, n_feats, kernel_size, groups, bias=True, bn=False, dilation = 1, act=nn.ReLU(True), res_scale=1):
        super(ResBlock_main, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, groups=groups, padding= dilation * (kernel_size-1)//2, dilation=dilation, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class DenseResBlockChain(nn.Module):
    def __init__(self, n_feats, kernel_size, n_blocks=4):
        super(DenseResBlockChain, self).__init__()
        self.blocks = nn.ModuleList([ResBlock_main(n_feats, kernel_size) for _ in range(n_blocks)])

    def forward(self, x):
        features = []
        out = x
        for block in self.blocks:
            out = block(out)  # ResBlock_main applies conv+relu+residual (+x)
            features.append(out)
        
        # concatenate all outputs along channel dimension
        dense_out = torch.cat(features, dim=1)
        return dense_out

class Block2(nn.Module): # has channel and spatial attention
    def __init__(self, channels, n_features=128*2):
        super(Block2, self).__init__()
        self.conv1 = nn.Conv2d(channels, n_features, 3, bias=True, padding='same')
        self.conv2 = nn.Conv2d(channels, n_features, 5, bias=True, padding='same')
        self.conv3 = nn.Conv2d(channels, n_features, 7, bias=True, padding='same')
        self.conv4 = nn.Conv2d(n_features * 3, n_features, 1, bias=True, padding=0)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x4 = torch.cat([x1, x2, x3], dim=1)
        x5 = F.relu(self.conv4(x4))
        return x5#output

class MultiScaleFE(nn.Module):
    def __init__(self, channels, features=128):
        super(MultiScaleFE, self).__init__()
        self.conv_1 = nn.Conv2d(channels, features, 3, padding=1)
        self.conv_2 = nn.Conv2d(channels, features, 3, padding=1)
        self.conv_3 = nn.Conv2d(channels, features, 3, padding=1)
        self.conv_4 = nn.Conv2d(channels, features, 3, padding=1)
        self.conv_5 = nn.Conv2d(channels, features, 3, padding=1)
        self.conv_6 = nn.Conv2d(channels, features, 3, padding=1)
        self.conv_7 = nn.Conv2d(channels + features * 6, features*2, 1, padding=0)
        self.conv_8 = nn.Conv2d(features*2, features*2, 1, padding=0)

    def forward(self, x):
        x1 = x
        up_1 = F.interpolate(x1, scale_factor=8, mode='bicubic', align_corners=False) #32 * 8
        up_1 = self.conv_1(up_1)
        up_2 = F.interpolate(x1, scale_factor=4, mode='bicubic', align_corners=False) #32 * 4
        up_2 = self.conv_2(up_2)
        up_3 = F.interpolate(x1, scale_factor=2, mode='bicubic', align_corners=False) #32 * 2
        up_3 = self.conv_3(up_3)
        down_1 = F.interpolate(x1, scale_factor=0.5, mode='bicubic', align_corners=False) #32 * 0.8
        down_1 = self.conv_4(down_1)
        down_2 = F.interpolate(x1, scale_factor=0.25, mode='bicubic', align_corners=False) #32 *0.25
        down_2 =self.conv_5(down_2)
        down_3 = F.interpolate(x1, scale_factor=0.75, mode='bicubic', align_corners=False) #32 * 0.75
        down_3 = self.conv_6(down_3)

        list = [up_1, up_2, up_3, down_1, down_2, down_3]
        resized_list = [x1]
        for i in list:
            if i.shape[2] == x.shape[2]:
                resized_list.append(i)
            else:
                resampled = F.interpolate(i, size=32, mode='bicubic', align_corners=False) #32 * 4
                resized_list.append(resampled)

        concat = torch.cat([x for x in resized_list], dim = 1)
        concat = self.conv_7(concat)
        concat = self.conv_8(concat)
        return concat

class SpectralLearner(nn.Module):
    def __init__(self):
        super(SpectralLearner, self).__init__()
        self.conv_16 = nn.Conv2d(256, 128, 1, padding=0)
        # self.conv_17 = nn.Conv2d(256, 512, 1, padding=0)
        # self.conv_18 = nn.Conv2d(512, 1284, 1, padding=0)
        # self.conv_19 = nn.Conv2d(1284, 128, 1, padding=0)

    def forward(self, x):
        x = F.silu(self.conv_16(x))
        # x = self.conv_17(x)
        # x = F.silu(self.conv_18(x))
        # x = self.conv_19(x)
        return x

class MultiDilationBlock(nn.Module):
    def __init__(self):
        super(MultiDilationBlock, self).__init__()
        self.conv_1 = nn.Conv2d(256, 128, 3, padding=3, dilation=3)
        # self.avg_pool_1 = nn.AdaptiveAvgPool2d()
        self.conv_2 = nn.Conv2d(256, 128, 3, padding=5, dilation=5)
        self.conv_3 = nn.Conv2d(256, 128, 3, padding=7, dilation=7)
        self.conv_4 = nn.Conv2d(256, 128, 3, padding=9, dilation=9)

        self.fuse = nn.Conv2d(128*4, 256, 1, padding=0)

    def forward(self,x):
        x1 = self.conv_1(x)
        # x1 = F.silu(x1)
        x2 = self.conv_2(x)
        # x2 = F.silu(x2)
        x3 = self.conv_3(x)
        # x3 = F.silu(x3)
        x4 = self.conv_4(x)
        # x4 = F.silu(x4)
        concat = torch.cat([x1, x2, x3, x4], dim = 1)
        y = self.fuse(concat)
        return y
    

import torchvision
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


#hewit leo
class SpatialSR(torch.nn.Module):
    def __init__(self):
        super(SpatialSR, self).__init__()

        self.conv_9by9 = nn.Conv2d(128, 256, 3, padding=1)

        self.dense_res_1 = DenseResBlock(256, 256, 3)
        self.dense_res_2 = DenseResBlock(256, 256, 3)
        self.dense_res_3 = DenseResBlock(128, 128, 3)

        self.pix_shuff_2 = Upsample_pixel_shuffle(kernel_size=1, padding=0, in_channels=128)

        self.conv_10 = nn.Conv2d(128, 128, 3, padding=1)
        # self.conv_10_1 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv_11 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv_13 = nn.Conv2d(128, 128, 3, padding=1)
        # self.conv_14 = nn.Conv2d(128, 256, 1, padding=0)

        self.conv_mix_a = nn.Conv2d(256, 128, 1,padding=0)
        # self.conv_mix_b = nn.Conv2d(128, 128, 3,padding=1)

        # self.ca_1 = CALayer(256)
        # self.ca_2 = CALayer(256)
        # self.ca_3 = CALayer(256)


    def forward(self, x):

        raw_x = x
        # x = self.conv_14(x)
        # spectra = self.spectral_fe(x)

        # y = self.conv_15(raw_x)
        # input = x
        # glob_skip_2x = F.interpolate(x, size=(x.shape[2]*2, x.shape[2]*2), mode='bicubic', align_corners=False)
        glob_skip = F.interpolate(raw_x, size=(x.shape[2]*2, x.shape[2]*2), mode='bicubic', align_corners=False)

        input = self.conv_9by9(x)
        # input = F.silu(input)
        # input = F.relu(input)################################################################silu

        res_1 = self.dense_res_1(input)
        # res_1 = self.conv_res1(res_1)

        res_2 = self.dense_res_2(res_1)
        # res_2 = self.conv_res2(res_2)

        res_123 = self.conv_mix_a(res_2)

        ps_1 = self.pix_shuff_2(res_123)

        res_3 = self.dense_res_3(ps_1)
        # res_3 = self.conv_res3(res_3)

        # concat_all_8 = torch.cat([res_1, res_2, res_3], dim=1)
        # concat_all_8 = self.conv_mix_a(res_2)
        # concat_all_8 = F.silu(concat_all_8)
        # concat_all_8 = F.relu(concat_all_8)

        # res_123 = self.conv_10(concat_all_8) ### it is skip of 1D input
        # res_123 = F.silu(res_123)######################################################silu
        res_123 = res_3 + glob_skip

        # ps_concat = self.conv_13(ps_1) + glob_skip
        conv_11 = self.conv_11(res_123)
        return conv_11




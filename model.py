"""
*Preliminary* pytorch implementation.

Networks for voxelmorph model

In general, these are fairly specific architectures that were designed for the presented papers.
However, the VoxelMorph concepts are not tied to a very particular architecture, and we 
encourage you to explore architectures that fit your needs. 
see e.g. more powerful unet function in https://github.com/adalca/neuron/blob/master/neuron/models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.autograd import Variable
from torch.distributions.normal import Normal
import torchvision
import numpy as np

from networks.correlation_package.correlation import Correlation

#########################################################
### Correlation network ###
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, times=1):
    return nn.Sequential(
                        nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding),
                        nn.BatchNorm2d(out_planes),
                        nn.LeakyReLU(0.1),
            )
    # layers = []
    # for _ in range(times):
    #     # layers.append(
    #     #     nn.Sequential(
    #     #         nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding),
    #     #         nn.LeakyReLU(0.1),
    #     #     )
    #     # )
    #     layers.append(
    #         nn.Sequential(
    #                     nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding),
    #                     nn.BatchNorm2d(out_planes),
    #                     nn.LeakyReLU(0.1),
    #         )
    #     )
    # return nn.Sequential(*layers)


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1, times=1):
    layers = []
    for _ in range(times):
        # layers.append(
        #     nn.Sequential(
        #         nn.Upsample(scale_factor=2),
        #         nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding),
        #     )
        # )
        layers.append(
            nn.Sequential(
                        nn.Upsample(scale_factor=2),
                        nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding),
                        nn.BatchNorm2d(out_planes),
                        nn.LeakyReLU(0.1),

            )
        )
    return nn.Sequential(*layers)


class ASFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [512, 256, 256]
        self.inter_dim = self.dim[self.level]
        if level==0:
            self.stride_level_1 = conv(256, self.inter_dim, 3, 2)
            self.stride_level_2 = conv(256, self.inter_dim, 3, 2)
            self.expand = conv(self.inter_dim, 1024, 3, 1)
        elif level==1:
            self.compress_level_0 = conv(512, self.inter_dim, 1, 1)
            self.stride_level_2 = conv(256, self.inter_dim, 3, 2)
            self.expand = conv(self.inter_dim, 512, 3, 1)
        elif level==2:
            self.compress_level_0 = conv(512, self.inter_dim, 1, 1)
            self.expand = conv(self.inter_dim, 256, 3, 1)

        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis = vis


    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level==0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter = nnf.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =nnf.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized =x_level_1
            level_2_resized =self.stride_level_2(x_level_2)
        elif self.level==2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =nnf.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_resized =nnf.interpolate(x_level_1, scale_factor=2, mode='nearest')
            level_2_resized =x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = nnf.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:,:,:]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FeatureL2Norm(nn.Module):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)

class Encoder_S(nn.Module):
    def __init__(self):
        super(Encoder_S, self).__init__()
        self.conv1 = conv(1, 8, 3, 2)
        self.conv1_a = conv(8, 8, 3, 1)
        self.conv2 = conv(8, 16, 3, 2)
        self.conv2_a = conv(16, 16, 3, 1)
        # self.conv1 = conv(1, 8, 7, 2, 3)
        # self.conv1_a = conv(8, 8, 7, 1, 3)
        # self.conv2 = conv(8, 16, 5, 2, 2)
        # self.conv2_a = conv(16, 16, 5, 1, 2)
        self.conv3 = conv(16, 32, 3, 2)
        self.conv3_a = conv(32, 32, 3, 1)
        self.conv4 = conv(32, 64, 3, 2)
        self.conv4_a = conv(64, 64, 3, 1)

        self.corr = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1)
        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)

        md = (2*4 + 1) ** 2
        # md = 128
        self.corr_refine4 = conv(md, 64, 3, 1)           # Dimentional reduction
        self.corr2feat4 = conv(64, 64, 3, 1, times=3)
        self.predict_flow4 = predict_flow(64)
        self.flow_refine4 = deconv(2, 2, 3, 1, times=1)

        # md = 64
        self.corr_refine3 = conv(md, 32, 3, 1)           # Dimentional reduction
        self.corr2feat3 = conv(130, 32, 3, 1, times=3)
        self.predict_flow3 = predict_flow(32)
        self.flow_refine3 = deconv(2, 2, 3, 1, times=1)

        # md = 32
        self.corr_refine2 = conv(md, 16, 3, 1)           # Dimentional reduction
        self.corr2feat2 = conv(66, 16, 3, 1, times=3)   # 32 32 2 32
        self.predict_flow2 = predict_flow(16)
        self.flow_refine2 = deconv(2, 2, 3, 1, times=1)

        # md = 16
        self.corr_refine1 = conv(md, 8, 3, 1)            # Dimentional reduction
        self.corr2feat1 = conv(34, 8, 3, 1, times=3)     # 16 8 2 16
        self.predict_flow1 = predict_flow(8)
        self.flow_refine1 = deconv(2, 2, 3, 1, times=1)

        self.warp_refine4 = deconv(64, 32, 3, 1, times=1)
        self.warp_refine3 = deconv(32, 16, 3, 1, times=1)
        self.warp_refine2 = deconv(16, 8, 3, 1, times=1)

        self.upfeat4 = deconv(64, 64, 3, 1)
        self.upfeat3 = deconv(32, 32, 3, 1)
        self.upfeat2 = deconv(16, 16, 3, 1)
        self.upfeat1 = deconv(8, 8, 3, 1)

        self.flow_refine_s = conv(2, 2, 3, 1)
        self.flow_refine_s2 = conv(2, 2, 3, 1)
        self.flow_refine_s3 = conv(2, 2, 3, 1)
        self.flow_refine_s4 = conv(2, 2, 3, 1)

        self.corr2feat0 = conv(10, 2, 3, 1) # 2 16

        # self.featurenorm = FeatureL2Norm()

        self.se_layer41 = SELayer(64)
        self.se_layer42 = SELayer(64)
        self.se_layer31 = SELayer(32)
        self.se_layer32 = SELayer(32)

        # self.se_layer21 = SELayer(16)
        # self.se_layer22 = SELayer(16)
        # self.se_layer11 = SELayer(8)
        # self.se_layer12 = SELayer(8)

        # self.image_refine = conv(1, 1, 3, 1, times=3) ###############################
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def warp(self, x, flow):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        return output # without mask
        # mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        # mask = nn.functional.grid_sample(mask, vgrid)
        #
        # mask[mask < 0.999] = 0
        # mask[mask > 0] = 1
        #
        # return output * mask

    def forward(self, mov, fix):
        f_list = []
        warp_list = []
        # f_t = self.conv1(mov)
        # print('Level 1 : ', f_t.size())
        f11 = self.conv1_a(self.conv1(mov))
        f12 = self.conv1_a(self.conv1(fix))
        # print('Level 1 : ', f11.size())

        f21 = self.conv2_a(self.conv2(f11))
        f22 = self.conv2_a(self.conv2(f12))
        # print('Level 2 : ', f21.size())

        f31 = self.conv3_a(self.conv3(f21))
        f32 = self.conv3_a(self.conv3(f22))
        # print('Level 3 : ', f31.size())

        f41 = self.conv4_a(self.conv4(f31))
        f42 = self.conv4_a(self.conv4(f32))
        # print('Level 4 : ', f41.size())

        # f41 = self.featurenorm(f41)  # Feature normalization
        # f42 = self.featurenorm(f42)  # Feature normalization
        f41 = self.se_layer41(f41)      # SE normalization
        f42 = self.se_layer42(f42)      # SE normalization
        # f4s = torch.cat([f41, f42], 1)
        # corr4 = self.corr_refine4(self.corr_activation(f4s))
        corr4 = self.corr_refine4(self.corr_activation(self.corr(f41, f42)))
        # corr4 = self.featurenorm(corr4)  # Feature normalization
        feat4 = self.corr2feat4(corr4)
        flow4 = self.predict_flow4(feat4)

        up_flow4 = self.flow_refine4(flow4)
        up_feat4 = self.upfeat4(feat4)
        warp4 = self.warp(f41, 0.3125 * flow4)
        warp4 = self.warp_refine4(warp4)
        warp_list.append(warp4)  # memorize warp results
        f_list.append(f32)  # memorize feature results

        # print(f32.size(), warp4.size())
        # f32 = self.featurenorm(f32)  # Feature normalization
        # warp4 = self.featurenorm(warp4)  # Feature normalization
        f32 = self.se_layer32(f32)      # SE normalization
        warp4 = self.se_layer31(warp4)      # SE normalization
        # f3s = torch.cat([f32, warp4], 1)
        # corr3 = self.corr_refine3(self.corr_activation(f3s))
        corr3 = self.corr_refine3(self.corr_activation(self.corr(f32, warp4)))
        # corr3 = self.featurenorm(corr3)   # Feature normalization
        feat_fusion3 = torch.cat([corr3, f32, up_flow4, up_feat4], dim=1)
        feat3 = self.corr2feat3(feat_fusion3)
        flow3 = self.predict_flow3(feat3)
        up_flow3 = self.flow_refine3(flow3)
        up_feat3 = self.upfeat3(feat3)
        warp3 = self.warp(f31, 0.625 * flow3)
        warp3 = self.warp_refine3(warp3)
        warp_list.append(warp3) # memorize warp results
        f_list.append(f22) # memorize feature results

        # f2s = torch.cat([f22, warp3], 1)
        # corr2 = self.corr_refine2(self.corr_activation(f2s))
        corr2 = self.corr_refine2(self.corr_activation(self.corr(f22, warp3)))
        feat_fusion2 = torch.cat([corr2, f22, up_flow3, up_feat3], dim=1)
        feat2 = self.corr2feat2(feat_fusion2)
        flow2 = self.predict_flow2(feat2)
        up_flow2 = self.flow_refine2(flow2)
        up_feat2 = self.upfeat2(feat2)
        warp2 = self.warp(f21, 1.25 * flow2)
        warp2 = self.warp_refine2(warp2)
        warp_list.append(warp2) # memorize warp results
        f_list.append(f11) # memorize feature results

        # f1s = torch.cat([f11, warp2], 1)
        # corr1 = self.corr_refine1(self.corr_activation(f1s))
        corr1 = self.corr_refine1(self.corr_activation(self.corr(f11, warp2)))
        feat_fusion1 = torch.cat([corr1, f12, up_flow2, up_feat2], dim=1)
        feat1 = self.corr2feat1(feat_fusion1)
        flow1 = self.predict_flow1(feat1)
        up_flow1 = self.flow_refine1(flow1)
        up_feat1 = self.upfeat1(feat1)
        feat_fusion0 = torch.cat([up_flow1, up_feat1], dim=1)
        flow = self.corr2feat0(feat_fusion0)

        flow = self.flow_refine_s4(self.flow_refine_s3(self.flow_refine_s2(self.flow_refine_s(flow))))
        # flow = self.flow_refine_s(flow)
        warped = self.warp(mov, flow)

        # warped = self.image_refine(warped) ############################################################
        return warped, flow, f_list, warp_list


class Encoder_MF(nn.Module):
    def __init__(self):
        super(Encoder_MF, self).__init__()
        self.Encoder = Encoder_S()
        self.optimizer = torch.optim.Adam(self.Encoder.parameters(), lr=1e-3, betas=(0.5, 0.999))
        self.iteration = 0

    def forward(self, mov, fix):
        return self.Encoder(mov, fix)

    def update_iter(self, idx):
        self.iteration = idx

    def get_cur_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

##########################################################################################
# class Combine_modules(nn.Module):
#     def __init__(self, vol_size, nf_enc, nf_dec):
#         super(Combine_modules, self).__init__()
#         vol_size = (512, 512)
#         nf_enc = [8, 16, 32, 32]
#         nf_dec = [32, 32, 16, 16, 16, 8, 8]
#         self.Voxelmorph = cvpr2018_net(vol_size, nf_enc, nf_dec)
#         self.Encoder_MF = Encoder_S()
#         self.spatial_transform = SpatialTransformer(vol_size)
#
#         self.fusion = conv(4, 2, 3, 1)
#         self.flow_refinement = conv(2, 2, 3, 1)
#         self.flow_refinement2 = conv(2, 2, 3, 1)
#         self.flow_refinement3 = conv(2, 2, 3, 1)
#         self.iteration = 0
#
#     def forward(self, mov, fix):
#         flow_MF, warped_MV = self.Encoder_MF(mov, fix)
#         flow_vox, warped_vox = self.Voxelmorph(mov, fix)
#         flow_fused = self.fusion(torch.cat([flow_MF, flow_vox], dim=1))
#
#         flow = self.flow_refinement(self.flow_refinement2(self.flow_refinement3(flow_fused)))
#         warp = self.spatial_transform(mov, flow)
#
#         return flow, warp
#
#     def update_iter(self, idx):
#         self.iteration = idx
#
#     def get_cur_lr(self, optimizer):
#         for param_group in optimizer.param_groups:
#             return param_group['lr']
#
#
# class Model_s2(nn.Module):
#     def __init__(self):
#         super(Model_s2, self).__init__()
#         self.model_s2 = Combine_modules()
#         self.optimizer = torch.optim.Adam(self.model_s2.parameters(), lr=1e-3, betas=(0.5, 0.999))
#         self.iteration = 0
#
#     def forward(self, mov, fix):
#         return self.model_s2(mov, fix)
#
#     def update_iter(self, idx):
#         self.iteration = idx
#
#     def get_cur_lr(self, optimizer):
#         for param_group in optimizer.param_groups:
#             return param_group['lr']

# if __name__ == '__main__':
#
#     import warnings
#     warnings.filterwarnings("ignore")
#
#     n = 512
#     x1 = torch.randn((4, 1, n, n)).cuda()
#     x2 = torch.randn((4, 1, n, n)).cuda()
#     # t = int(n/8)
#     # gt = torch.randn((4, 1, t, t)).cuda()
#     gt = torch.randn((4, 1, n, n)).cuda()
#
#
#     model = Encoder_S().cuda()
#
#     model = torch.nn.DataParallel(model, device_ids=[0, 1])
#
#     criterion = nn.MSELoss()
#     optim = torch.optim.Adam(model.module.parameters(), lr=1e-3, weight_decay=5e-5)
#
#     num_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
#     print('The number of parameters of Corr : ', num_params)
#
#     loss = 100
#     while loss > 0.8:
#         out = model(x1, x2)
#
#         optim.zero_grad()
#         loss = criterion(out, gt)
#         print('Loss:', loss.item())
#         loss.backward()
#         optim.step()

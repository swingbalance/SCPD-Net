"""
*Preliminary* pytorch implementation.

Losses for VoxelMorph
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from skimage.feature import match_template
import pytorch_ssim
from collections import namedtuple

import torch
import torchvision.models.vgg as vgg
from Seg_loss.lovasz_loss import LovaszSoftmax

def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

    if penalty == 'l2':
        dy = dy * dy
        dx = dx * dx

    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0


def mse_loss(x, y):
    return torch.mean((x - y) ** 2)

# class seg_loss(torch.nn.Module):
#     def __init__(self):
#         super(seg_loss, self).__init__()
#         self.seg_loss = LovaszSoftmax()
#     def forward(self, x, y):
#         return self.seg_loss(x, y)

def diceLoss(pred, target):
    """
    This definition generalize to real valued pred and target vector. This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    dice = ((2. * intersection + smooth) / (A_sum + B_sum + smooth))
    return 1-dice
    # return 1 -

# def diceLoss(y_true, y_pred, nc=2):
#     # Original dice loss
#     # dice = 0
#     # smooth = 1.0
#     #
#     # for k in range(0, nc):
#     #     # print(k)
#     #     index_true = (y_true == k).float()
#     #     index_pred = (y_pred == k).float()
#     #     top = 2*torch.sum(index_true * index_pred, dim=[1, 2, 3])
#     #     bottom = torch.sum(index_true + index_pred, dim=[1, 2, 3])
#     #     dice += torch.mean(1 - (top + smooth)/(bottom + smooth))
#     #
#     # return dice
#     y_true = y_true / 2.0
#     y_pred = y_pred / 2.0
#     # epic = 1
#     top = 2 * torch.mul(y_true, y_pred, [1, 2, 3]).sum()
#     bottom = (y_true + y_pred, [1, 2, 3]).sum()
#     dice = torch.mean(top / bottom)
#     return -dice


def ncc_loss(I, J, win=None):
    """
    calculate the normalize cross correlation between I and J
    assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    """

    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims

    conv_fn = getattr(F, 'conv%dd' % ndims)
    I2 = I * I
    J2 = J * J
    IJ = I * J

    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross * cross / (I_var * J_var + 1e-5)

    # return -1 * torch.mean(cc)
    return 1 - ((torch.mean(cc) + 1.0) / 2.0)

def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def ski_ncc(src, tgt):
    src_np = src[0, 0, :, :].detach().cpu().numpy()
    tgt_np = tgt[0, 0, :, :].detach().cpu().numpy()
    return torch.from_numpy(match_template(src_np, tgt_np))


def ssim_loss(img, tem):
    ssim_fn = pytorch_ssim.SSIM(window_size=11)
    return 1.0 - ssim_fn(img, tem)


def img_grad_loss(src, tgt):

    src_x = torch.pow(src[:, :, 1:, :] - src[:, :, :-1, :], 2)
    src_y = torch.pow(src[:, :, :, 1:] - src[:, :, :, :-1], 2)
    tgt_x = torch.pow(tgt[:, :, 1:, :] - tgt[:, :, :-1, :], 2)
    tgt_y = torch.pow(src[:, :, :, 1:] - tgt[:, :, :, :-1], 2)

    return torch.mean(tgt_x - src_x) + torch.mean(tgt_y - src_y)

# VGG net
# LossOutput = namedtuple(
#     "LossOutput", ["relu1", "relu2", "relu3", "relu4", "relu5"])
#
# class LossNetwork(torch.nn.Module):
#     """Reference:
#         https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
#     """
#
#     def __init__(self):
#         super(LossNetwork, self).__init__()
#         self.vgg_layers = vgg.vgg19(pretrained=True).features
#         self.layers_map = {'relu4_2': '22', 'relu2_2': '8', 'relu3_2': '13','relu1_2': '4'}
#         # self.layer_name_mapping = {
#         #     '3': "relu1",
#         #     '8': "relu2",
#         #     '17': "relu3",
#         #     '26': "relu4",
#         #     '35': "relu5",
#         # }
#
#     def forward(self, x):
#         # output = {}
#         # for name, module in self.vgg_layers._modules.items():
#         #     x = module(x)
#         #     if name in self.layer_name_mapping:
#         #         output[self.layer_name_mapping[name]] = x
#         # return LossOutput(**output)
#
#         outputs = []
#         for name, module in self.vgg_layers._modules.items():
#             x = module(x)
#             if name in self.layers_map:
#                 outputs += [x]
#         return outputs + [x]


class VGGFeatureExtractor(torch.nn.Module):
    # Extract features from intermediate layers of a network

    def __init__(self, submodule, extracted_layers):
        super(VGGFeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]
        return outputs + [x]

''' '''
def EPE(input_flow, target_flow, sparse=False, mean=True, sum=False):

    EPE_map = torch.norm(target_flow-input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    elif sum:
        return EPE_map.sum()
    else:
        return EPE_map.sum()/batch_size

def L1_loss(input_flow, target_flow):
    L1 = torch.abs(input_flow-target_flow)
    L1 = torch.sum(L1, 1)
    return L1


def L1_charbonnier_loss(input_flow, target_flow, sparse=False, mean=True, sum=False):

    batch_size = input_flow.size(0)
    epsilon = 0.01
    alpha = 0.4
    L1 = L1_loss(input_flow, target_flow)
    norm = torch.pow(L1 + epsilon, alpha)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        norm = norm[~mask]
    if mean:
        return norm.mean()
    elif sum:
        return norm.sum()
    else:
        return norm.sum()/batch_size

def sparse_max_pool(input, size):
    '''Downsample the input by considering 0 values as invalid.
    Unfortunately, no generic interpolation mode can resize a sparse map correctly,
    the strategy here is to use max pooling for positive values and "min pooling"
    for negative values, the two results are then summed.
    This technique allows sparsity to be minized, contrary to nearest interpolation,
    which could potentially lose information for isolated data points.'''

    positive = (input > 0).float()
    negative = (input < 0).float()
    output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
    return output

# def multiscaleEPE(network_output, target_flow, robust_L1_loss=False, mask=None, weights=None,
#                   sparse=False, mean=False):
#     '''
#     here the ground truth flow is given at the higest resolution and it is just interpolated
#     at the different sized (without rescaling it)
#     :param network_output:
#     :param target_flow:
#     :param weights:
#     :param sparse:
#     :return:
#     '''
#
#     def one_scale(output, target, sparse, robust_L1_loss=False, mask=None, mean=False):
#         b, _, h, w = output.size()
#         print(output.shape)
#         print(target.shape)
#         if sparse:
#             target_scaled = sparse_max_pool(target, (h, w))
#
#             if mask is not None:
#                 mask = sparse_max_pool(mask.float().unsqueeze(1), (h, w))
#         else:
#             target_scaled = F.interpolate(target, (h, w), mode='bilinear')
#
#             if mask is not None:
#                 # mask can be byte or float or uint8 or int
#                 # resize first in float, and then convert to byte/int to remove the borders which are values between 0 and 1
#                 mask = F.interpolate(mask.float().unsqueeze(1), (h, w), mode='bilinear').byte()
#
#         if robust_L1_loss:
#             if mask is not None:
#                 return L1_charbonnier_loss(output * mask.float(), target_scaled * mask.float(), sparse, mean=mean, sum=False)
#             else:
#                 return L1_charbonnier_loss(output, target_scaled, sparse, mean=mean, sum=False)
#         else:
#             if mask is not None:
#                 return EPE(output * mask.float(), target_scaled * mask.float(), sparse, mean=mean, sum=False)
#             else:
#                 return EPE(output, target_scaled, sparse, mean=mean, sum=False)
#
#     if type(network_output) not in [tuple, list]:
#         network_output = [network_output]
#     if weights is None:
#         weights = [0.32, 0.08, 0.02, 0.01, 0.005]  # as in original article
#     assert(len(weights) == len(network_output))
#
#     loss = 0
#     for output, weight in zip(network_output, weights):
#         # from smallest size to biggest size (last one is a quarter of input image size
#         loss += weight * one_scale(output, target_flow, sparse, robust_L1_loss=robust_L1_loss, mask=mask, mean=mean)
#     return loss


def multiscale_loss(network_output, target_flow, weights=None):
    def one_scale(output, target):
        b, _, h, w = output.size()
        return L1_charbonnier_loss(output, target, mean=True, sum=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.32, 0.08, 0.02, 0.01, 0.005]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, gt, weight in zip(network_output, target_flow, weights):
        # from smallest size to biggest size (last one is a quarter of input image size
        loss += weight * one_scale(output, gt)
    return loss


def sobel(window_size):
    assert(window_size%2!=0)
    ind=int(window_size/2)
    matx=[]
    maty=[]
    for j in range(-ind,ind+1):
        row=[]
        for i in range(-ind,ind+1):
            if (i*i+j*j)==0:
                gx_ij=0
            else:
                gx_ij=i/float(i*i+j*j)
            row.append(gx_ij)
        matx.append(row)
    for j in range(-ind,ind+1):
        row=[]
        for i in range(-ind,ind+1):
            if (i*i+j*j)==0:
                gy_ij=0
            else:
                gy_ij=j/float(i*i+j*j)
            row.append(gy_ij)
        maty.append(row)

    # matx=[[-3, 0,+3],
    # 	  [-10, 0 ,+10],
    # 	  [-3, 0,+3]]
    # maty=[[-3, -10,-3],
    # 	  [0, 0 ,0],
    # 	  [3, 10,3]]
    if window_size==3:
        mult=2
    elif window_size==5:
        mult=20
    elif window_size==7:
        mult=780

    matx=np.array(matx)*mult
    maty=np.array(maty)*mult

    return torch.Tensor(matx), torch.Tensor(maty)

def create_window(window_size, channel):
    windowx, windowy = sobel(window_size)
    windowx, windowy = windowx.unsqueeze(0).unsqueeze(0), windowy.unsqueeze(0).unsqueeze(0)
    windowx = torch.Tensor(windowx.expand(channel, 1, window_size, window_size))
    windowy = torch.Tensor(windowy.expand(channel, 1, window_size, window_size))

    return windowx, windowy


def gradient(img, windowx, windowy, window_size, padding, channel):
    # print(np.unique(img.detach().cpu().numpy()))
    if channel > 1:  # do convolutions on each channel separately and then concatenate
        gradx = torch.ones(img.shape)
        grady = torch.ones(img.shape)
        for i in range(channel):
            gradx[:, i, :, :] = F.conv2d(img[:, i, :, :].unsqueeze(0), windowx, padding=padding, groups=1).squeeze(
                0)  # fix the padding according to the kernel size
            grady[:, i, :, :] = F.conv2d(img[:, i, :, :].unsqueeze(0), windowy, padding=padding, groups=1).squeeze(0)

    else:
        # print('XXXX')
        gradx = F.conv2d(img, windowx, padding=padding, groups=1)
        grady = F.conv2d(img, windowy, padding=padding, groups=1)
    # print('-------')
    # print(np.unique(grady.detach().cpu().numpy()))
    # print('-------')
    return gradx, grady


class SobelGrad(torch.nn.Module):
    def __init__(self, window_size=3, padding=1):
        super(SobelGrad, self).__init__()
        self.window_size = window_size
        self.padding = padding
        self.channel = 1  # out channel
        self.windowx, self.windowy = create_window(window_size, self.channel)

    def forward(self, pred, label):
        (batch_size, channel, _, _) = pred.size()
        if pred.is_cuda:
            self.windowx = self.windowx.cuda(pred.get_device())
            self.windowx = self.windowx.type_as(pred)
            self.windowy = self.windowy.cuda(pred.get_device())
            self.windowy = self.windowy.type_as(pred)
        # print(self.windowx)
        pred_gradx, pred_grady = gradient(pred, self.windowx, self.windowy, self.window_size, self.padding, channel)
        label_gradx, label_grady = gradient(label, self.windowx, self.windowy, self.window_size, self.padding, channel)

        return pred_gradx, pred_grady, label_gradx, label_grady
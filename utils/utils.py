import os
import shutil

import torch
import numpy as np
# from skimage.exposure import match_histograms
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import hist_match

def his_match(input_tensor, target_tensor):
    n, c, h, w = input_tensor.size()
    matched = []
    input_tensor_t = input_tensor.detach().cpu().numpy()
    target_tensor_t = target_tensor.detach().cpu().numpy()
    input_tensor_t = (input_tensor_t * 255.0).astype(np.uint8)
    target_tensor_t = (target_tensor_t * 255.0).astype(np.uint8)

    for i in range(n):
        s = np.squeeze(input_tensor_t[i])
        t = np.squeeze(target_tensor_t[i])
        m = hist_match.hist_match_polynomial(s, t)

        matched.append(np.expand_dims(m, 0))

    matched = np.asarray(matched).astype(np.float32) / 255.0

    return torch.from_numpy(matched)

class GramMatrix(torch.nn.Module):
    def __init__(self):
        super(GramMatrix, self).__init__()

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a, b, c * d)  # resize F_XL into \hat F_XL

        G = torch.bmm(features, features.transpose(1, 2))  # compute the gram product

        # normalize the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(b * c * d)

def warper(x, flow):
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
    vgrid = torch.autograd.Variable(grid) + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(x.float(), vgrid)
    return output # without mask

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def check_and_create_dir(path):
    try:
        os.makedirs(path)
        print('\tCreate {:s} successful'.format(path))
    except IsADirectoryError:
        print("\t{:s} exists ".format(path))
    except FileExistsError:
        print("\t{:s} exists ".format(path))
    except PermissionError:
        print("\tPermission denied")

def recreate_specific_dir(path):
    if os.path.isdir(path):
        print('Deleting {:s}'.format(path))
        shutil.rmtree(path)
    print('Recreating {:s} successful'.format(path))
    os.makedirs(path)

def create_exp_env(exp_dir, exp_name):
    folder_list = ['checkpoint', 'output', 'config', 'logger']

    exp_base = os.path.join(os.getcwd(), exp_dir, exp_name) # os.getcwd() -> get current dir

    print('Creating experiment "{:s}"'.format(exp_name))
    check_and_create_dir(exp_base)

    folder_dict = dict()
    for folder_name in folder_list:
        exp_temp = os.path.join(exp_base, folder_name)
        check_and_create_dir(exp_temp)
        folder_dict[folder_name] = exp_temp

    recreate_specific_dir(folder_dict['logger'])
    recreate_specific_dir(folder_dict['output'])
    return folder_dict

def save_model_information(exp_dict, model, mode='state_dict'):
    if mode == 'state_dict':
        save_file_name = os.path.join(exp_dict['checkpoint'], 'model.pth')

        ckpt_s2 = {'iteration': model.module.iteration,
                   'state_dict': model.module.Encoder.state_dict(),
                   'optimizer': model.module.optimizer.state_dict(),
                   }
        torch.save(ckpt_s2, save_file_name)

    elif mode == 'params_num':
        exp_model_note_dir = os.path.join(exp_dict['logger'], 'Model_params_and_Optimizer.txt')
        with open(exp_model_note_dir, 'w') as f:
            num_params = sum(p.numel() for p in model.module.Encoder.parameters() if p.requires_grad)
            print('The number of parameters of model_s2 : ', num_params)
            f.writelines('The number of parameters of model_s2 : {:d}\n'.format(num_params))

            f.writelines('Optimizer_G : {:s}\n'.format(str(model.module.optimizer.state_dict())))

    elif mode == 'model_py_file':
        src = os.path.join(os.getcwd(), 'model.py')
        dst = os.path.join(exp_dict['checkpoint'], 'model.py')

        if os.path.isfile(dst):
            os.remove(dst)
            print('Remove model python file at {:s}'.format(dst))
        shutil.copyfile(src, dst)
        print('Copy model file from {:s} to {:s} successful'.format(src, dst))

        src = os.path.join(os.getcwd(), 'Config.py')
        dst = os.path.join(exp_dict['config'], 'config.py')

        if os.path.isfile(dst):
            os.remove(dst)
            print('Remove config python file at {:s}'.format(dst))
        shutil.copyfile(src, dst)
        print('Copy config file from {:s} to {:s} successful'.format(src, dst))

def gif_maker(src1, src2):
    # Choose first tensor per batch
    return torch.cat([src1[0].unsqueeze(0).unsqueeze(0), src2[0].unsqueeze(0).unsqueeze(0)], dim=1)

def state_dict_path(config):
    path = os.path.join(config.exp_dir, config.exp_name, 'checkpoint')
    fname = os.path.join(path, 'model.pth')
    print('Load checkpoint from {:s}'.format(fname))
    return fname

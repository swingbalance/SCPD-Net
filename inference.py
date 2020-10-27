# Python imports
import os
import warnings
warnings.filterwarnings("ignore")
import time
import shutil


# External imports
import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import imageio as imgio
import matplotlib.pyplot as plt
import cv2
import importlib

# Internal imports
# from model import Encoder_MF
# from experiment.encoder_mf_v10.checkpoint.model import Encoder_MF
import losses
from dataloader.Pair_NIH_Dataloader import Pair_NIH_Dataloader
from Config import Config_testing
from utils.utils import *
import metrics

from torch.utils.tensorboard import SummaryWriter
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    config_testing = Config_testing()
    print('Testing on', config_testing.exp_name)

    dataloader_testing = Pair_NIH_Dataloader(config_testing, train=False)
    print('Evaluate on %d pairs'%(config_testing.batch_size * dataloader_testing.__len__()))

    model_path = '.'.join([config_testing.exp_dir, config_testing.exp_name, 'checkpoint', 'model'])
    module = importlib.import_module(model_path)
    # 1. Load model checkpoint for multi-GPU testing
    model = module.Encoder_MF().cuda()
    model = torch.nn.DataParallel(model, device_ids=config_testing.gpu_ids)

    # 2. Load checkpoint
    # Note that my model was trained on multi-GPU
    ckpt_path = os.path.join(config_testing.exp_dir, config_testing.exp_name, 'checkpoint', 'model_latest.pth')
    print('Load model from', ckpt_path)
    ckpt = torch.load(ckpt_path)
    model.module.Encoder.load_state_dict(ckpt['state_dict'])
    model.module.eval()

    # 2. Load model checkpoint for single GPU testing
    # model = Encoder_MF().cuda()
    # ckpt = torch.load(mpath)
    # model.Encoder.load_state_dict(ckpt['state_dict'])
    # model.eval()

    # 3. Load model checkpoint for CPU testing
    # model = Encoder_MF()
    # ckpt = torch.load(mpath, map_location=lambda storage, loc: storage)
    # model.Encoder.load_state_dict(ckpt['state_dict'])
    # model.eval()

    # results_path = os.path.join(config_testing.exp_dir, config_testing.exp_name, 'results')
    results_path = os.path.join(config_testing.exp_dir, config_testing.exp_name, 'results_gif')

    check_and_create_dir(results_path)

    start_time = time.time()
    cnt_pairs = 0
    IoU = AverageMeter('IoU', ':.4e')
    MAD = AverageMeter('MAD', ':.4e')
    MSE = AverageMeter('MSE', ':.4e')
    PSNR = AverageMeter('PSNR', ':.4e')
    SSIM = AverageMeter('SSIM', ':.4e')
    NCC = AverageMeter('NCC', ':.4e')

    batch_start_time = time.time()
    for idx, data in enumerate(dataloader_testing):
        ' Load images '
        if config_testing.seg :
            input_moving_crop, input_fixed_crop, input_moving_seg, input_fixed_seg, ID_s, ID_t = data
            input_moving_seg, input_fixed_seg = input_moving_seg.cuda(), input_fixed_seg.cuda()
        else:
            input_moving_crop, input_fixed_crop, ID_s, ID_t = data
        input_moving_crop, input_fixed_crop = input_moving_crop.cuda(), input_fixed_crop.cuda()

        ' Histogram matching : moving image -> fixed image '
        # input_moving_hm = his_match(input_moving_crop, input_fixed_crop)
        # s = time.time()
        # input_moving_hm = his_match(input_moving_crop, input_fixed_crop).cuda()
        # for i in range(input_moving_hm.size(0)):
        #     moving_hm = input_moving_hm[i, 0, :, :].detach().cpu().numpy()
        #     moving_hm = (moving_hm * 255.0).astype(np.uint8)
        #     cv2.imwrite('data_crop/moving_hm/' + ID_s[i], moving_hm)
        # print('%.4f'%(time.time() - s))
        ' Predict by model '
        warped, flow, _, _ = model(input_moving_crop, input_fixed_crop)
        if config_testing.seg:
            warped_seg = warper(input_moving_seg, flow).round()

        IoU.update(metrics.dice(warped_seg, input_fixed_seg).item())
        MAD.update(metrics.mad(warped, input_fixed_crop).item())
        MSE.update(metrics.mse(warped, input_fixed_crop).item())
        PSNR.update(metrics.psnr(warped, input_fixed_crop).item())
        if config_testing.vox:
            NCC.update(metrics.ncc(warped.float(), input_fixed_crop.float()).item())
            SSIM.update(metrics.ssim(warped.float(), input_fixed_crop.float()).item())
        else:
            NCC.update(metrics.ncc(warped, input_fixed_crop).item())
            SSIM.update(metrics.ssim(warped, input_fixed_crop).item())

        for i in range(input_moving_crop.size(0)):

            # mov_ori = np.squeeze(input_moving_crop[i].detach().cpu().numpy())
            mov = np.squeeze(input_moving_crop[i].detach().cpu().numpy())
            fix = np.squeeze(input_fixed_crop[i].detach().cpu().numpy())
            warp = np.squeeze(warped[i].detach().cpu().numpy())
            if config_testing.seg:
                mov_seg = np.squeeze(input_moving_seg[i].detach().cpu().numpy())
                fix_seg = np.squeeze(input_fixed_seg[i].detach().cpu().numpy())
                warp_seg = np.squeeze(warped_seg[i].detach().cpu().numpy())

            mov_id = str(ID_s[i]).split('.')[0]
            fix_id = str(ID_t[i]).split('.')[0]
            img_path = os.path.join(results_path, mov_id)
            check_and_create_dir(img_path)

            mov_path_ori = os.path.join(img_path, mov_id + '_O.png')
            mov_path = os.path.join(img_path, mov_id + '_M.png')
            fix_path = os.path.join(img_path, fix_id + '_F.png')
            warp_path = os.path.join(img_path, mov_id + '_W.png')
            if config_testing.seg:
                mov_seg_path = os.path.join(img_path, mov_id + '_Mseg.png')
                fix_seg_path = os.path.join(img_path, fix_id + '_Fseg.png')
                warp_seg_path = os.path.join(img_path, mov_id + '_Wseg.png')

            # plt.imsave(mov_path_ori, mov_ori, cmap='gray')
            plt.imsave(mov_path, mov, cmap='gray')
            plt.imsave(fix_path, fix, cmap='gray')
            plt.imsave(warp_path, warp, cmap='gray')
            if config_testing.seg:
                plt.imsave(mov_seg_path, mov_seg, cmap='gray')
                plt.imsave(fix_seg_path, fix_seg, cmap='gray')
                plt.imsave(warp_seg_path, warp_seg, cmap='gray')

            ' Make GIF '
            mov = (mov * 255.0).astype(np.uint8)
            warp = (warp * 255.0).astype(np.uint8)
            fix = (fix * 255.0).astype(np.uint8)
            mov2warp = gif_maker(mov, warp)
            warp2fix = gif_maker(warp, fix)
            mov2warp_path = os.path.join(img_path, 'mov2warp.gif')
            warp2fix_path = os.path.join(img_path, 'warp2fix.gif')
            imgio.mimsave(mov2warp_path, mov2warp, duration=0.6)
            imgio.mimsave(warp2fix_path, warp2fix, duration=0.6)

            if config_testing.seg:
                mov_seg = (mov_seg / 2.0 * 255.0).astype(np.uint8)
                fix_seg = (fix_seg / 2.0 * 255.0).astype(np.uint8)
                warp_seg = (warp_seg / 2.0 * 255.0).astype(np.uint8)

                img = np.zeros((512, 512, 3), dtype=np.uint8)

                ret, binary = cv2.threshold(mov_seg, 100, 255, cv2.THRESH_BINARY)
                binary, contours_m, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours_m, -1, (255, 0, 0), 1) #Blue

                ret, binary = cv2.threshold(fix_seg, 100, 255, cv2.THRESH_BINARY)
                binary, contours_f, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours_f, -1, (0, 255, 0), 1) #Green

                ret, binary = cv2.threshold(warp_seg, 100, 255, cv2.THRESH_BINARY)
                binary, contours_w, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours_w, -1, (0, 0, 255), 1) #Red

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                contour_path = os.path.join(img_path, 'contour.png')
                plt.imsave(contour_path, img)

                cv2.putText(mov, 'Mov', (500, 240), cv2.FONT_HERSHEY_SIMPLEX,
                            1, 255, 3, cv2.LINE_AA)
                cv2.putText(warp, 'Warp', (500, 240), cv2.FONT_HERSHEY_SIMPLEX,
                            1, 255, 3, cv2.LINE_AA)
                cv2.putText(fix, 'Fix', (500, 240), cv2.FONT_HERSHEY_SIMPLEX,
                            1, 255, 3, cv2.LINE_AA)

                mov2warp_seg = gif_maker(mov_seg, warp_seg)
                warp2fix_seg = gif_maker(warp_seg, fix_seg)
                m2w2f = []
                m2w2f.append(mov_seg)
                m2w2f.append(warp_seg)
                m2w2f.append(fix_seg)

                mov2warp_seg_path = os.path.join(img_path, 'mov2warp_seg.gif')
                warp2fix_seg_path = os.path.join(img_path, 'warp2fix_seg.gif')
                m2w2f_path = os.path.join(img_path, 'm2w2f.gif')
                imgio.mimsave(mov2warp_seg_path, mov2warp_seg, duration=0.6)
                imgio.mimsave(warp2fix_seg_path, warp2fix_seg, duration=0.6)
                imgio.mimsave(m2w2f_path, m2w2f, duration=0.6)

            batch_end_time = time.time()
            print('\t%d\t%.4f secs\n'%(cnt_pairs, batch_end_time - batch_start_time))
            batch_start_time = time.time()
            cnt_pairs = cnt_pairs + 1
    end_time = time.time()
    print('Inference on {:s}, {:d} pairs, spends {:.4f} secs'.format(config_testing.exp_name,
                                                                     cnt_pairs,
                                                                     end_time - start_time))

    line_IoU = 'Averaged IoU : ' + str(IoU.avg)
    line_MAD = 'Averaged MAD : ' + str(MAD.avg)
    line_MSE = 'Averaged MSE : ' + str(MSE.avg)
    line = '\n'.join([line_IoU, line_MAD, line_MSE])
    print(line)

    with open(os.path.join(config_testing.exp_dir, config_testing.exp_name, 'eval.txt'), 'w') as f:
        f.writelines(line)




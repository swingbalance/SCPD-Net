# Python imports
import os
import warnings
warnings.filterwarnings("ignore")
import time

# External imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models

# Internal imports
from model import Encoder_MF
import losses
from dataloader.Pair_NIH_Dataloader import Pair_NIH_Dataloader
from Config import Config
from utils.utils import *
import metrics

from torch.utils.tensorboard import SummaryWriter
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
torch.backends.cudnn.benchmark = True
from torchsummary import summary

def run():
    torch.multiprocessing.freeze_support()
    print('Loop start .....')

def tf_board_process(log_dir):
    os.system('tensorboard --logdir ' + log_dir)
    print('Tensorboard started......')

def train(config, exp_fold_dict):
    writer = SummaryWriter(log_dir=exp_fold_dict['logger'])

    model = Encoder_MF().cuda()

    if len(config.gpu_ids) and not config.fp16:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        optimizer = model.module.optimizer
    else:
        from apex import amp
        print('Using apex to accelerate training.........')
        model, [optimizer] = amp.initialize(model, [model.optimizer], opt_level='O1')
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
    model.module.train()
    # summary(model=model, input_size=[(1, 512, 512), (1, 512, 512)])
    save_model_information(exp_fold_dict, model, mode='params_num')
    save_model_information(exp_fold_dict, model, mode='model_py_file')

    # Losses
    # Similarity loss
    sim_loss_fn = losses.ncc_loss

    # Gradient loss
    grad_loss_fn = losses.gradient_loss

    # VGG loss
    layers_map = {'relu5': '35', 'relu4': '26', 'relu3': '17', 'relu2': '8', 'relu1': '3'}
    vgg19 = models.vgg19(pretrained=True)
    vgg19.cuda()
    vgg19 = torch.nn.DataParallel(vgg19, device_ids=config.gpu_ids)
    vgg19.eval()
    vgg = losses.VGGFeatureExtractor(vgg19.module.features, [layers_map['relu1'], layers_map['relu2'],
                                                             layers_map['relu3'], layers_map['relu4'],
                                                             layers_map['relu5']])
    distance_fn = torch.nn.MSELoss()  # L2 loss for perceptual loss

    # Dataloader
    dataloader = Pair_NIH_Dataloader(config)

    start_time = time.time()

    loss_best = 1e5
    iter_best = 0
    start = time.time()

    # Training loop.
    for idx in range(config.n_iter):
        model.module.update_iter(idx)

        # If the cropped image has been saved, use this for training acceleration
        if config.seg:
            input_moving_crop, input_fixed_crop, input_moving_seg, input_fixed_seg, ID_s, ID_t = next(iter(dataloader))
            input_moving_seg, input_fixed_seg = input_moving_seg.cuda(), input_fixed_seg.cuda()
        else:
            input_moving_crop, input_fixed_crop, ID_s, ID_t = next(iter(dataloader))

        input_moving_crop, input_fixed_crop = input_moving_crop.cuda(), input_fixed_crop.cuda()

        fname = str(ID_s).split('\'')[1] + ', ' + str(ID_t).split('\'')[1]

        ' Histogram matching : moving image -> fixed image '
        input_moving_hm = input_moving_crop
        # input_moving_hm = his_match(input_moving_crop, input_fixed_crop).cuda()

        ''' Stage2: To produce warped image and flow field '''
        warped, flow, flow_warp, flow_gt = model(input_moving_hm, input_fixed_crop)

        ' Calculate loss '
        loss_sim = sim_loss_fn(warped, input_fixed_crop)  # NCC
        loss_flow = grad_loss_fn(flow)

        warp_vgg = vgg(warped.repeat(1, 3, 1, 1))
        fix_vgg = vgg(input_fixed_crop.repeat(1, 3, 1, 1))
        loss_vgg = 0
        for m in range(len(warp_vgg)):
            loss_vgg += distance_fn(warp_vgg[m], fix_vgg[m])

        loss = loss_sim + loss_flow + loss_vgg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ' Loss filter for visualization: Comment for ignored loss '
        loss_dict = {'Total': loss.item(),
                     'Sim': loss_sim.item(),
                     'Reg': loss_flow.item(),
                     'VGG': loss_vgg.item(),
                     }

        ' Evaluation '
        if (idx + 1) % config.n_upload_iter == 0:
            eval_mse = metrics.mse(warped, input_fixed_crop)
            eval_mad = metrics.mad(warped, input_fixed_crop)

            # Scalars
            writer.add_scalar('Evaluation_metric/intensity_based/MSE', eval_mse.float(), idx)
            writer.add_scalar('Evaluation_metric/intensity_based/MAD', eval_mad.float(), idx)

            # Loss curve
            writer.add_scalars('Loss/Encoder_MF', loss_dict, idx)

            # Images
            writer.add_image('Image/Input/Fixed', input_fixed_crop[0], idx)
            writer.add_image('Image/Input/Moving_histogram_matching', input_moving_hm[0], idx)
            writer.add_image('Image/Output/Warped_image', warped[0], idx)

            # GIF
            gif_mov2warp = gif_maker(input_moving_hm, warped)
            gif_warp2fix = gif_maker(warped, input_fixed_crop)
            gif_mov2fix = gif_maker(input_moving_hm, input_fixed_crop)
            gif_fix2fix = gif_maker(input_fixed_crop, input_fixed_crop)

            writer.add_video('GIF/1/Mov2fix', gif_mov2fix, fps=2, global_step=idx)
            writer.add_video('GIF/2/Fix2fix', gif_fix2fix, fps=2, global_step=idx)
            writer.add_video('GIF/3/Mov2warp', gif_mov2warp, fps=2, global_step=idx)
            writer.add_video('GIF/4/Warp2fix', gif_warp2fix, fps=2, global_step=idx)

        # if torch.isnan(loss):
        #     print('Nan at iter {:d}, break training!'.format(idx))
        #     break

        ' Memorize best loss'
        if loss_best > loss:
            ' Save checkpoint of best model '
            save_model_information(exp_fold_dict, model, mode='state_dict')
            print('------> Iter {:d}, Model saved'.format(idx))
            iter_best = idx
            loss_best = loss

        ' Display loss on terminal '' Save model weight and image results '
        if (idx+1) % config.n_display_iter == 0:
            print('\n\nIter %d ' % (idx+1))
            print('\tEncoder_MF:\n\t\t', end='')
            for name in loss_dict.keys():
                el = '\n' if name == list(loss_dict.keys())[-1] else ''  # print \n after last key
                print('{:s}: {:.4f}, '.format(name, loss_dict[name]), end=el)

            stage2_time = time.time()
            print('\tTime: %.4f secs\n' % (stage2_time - start_time))
            print('Best model checkpoint saved at {:d}'.format(iter_best))
            start_time = time.time()

        if (idx + 1) % config.n_save_iter == 0:
            print('Save checkpoint at iter {:d}k'.format(int((idx+1)/1000)))
            ' Save checkpoint '
            save_model_path = os.path.join(exp_fold_dict['checkpoint'],
                                           'model_{:d}k.pth'.format(int((idx+1)/1000)))
            ckpt_s2 = {'iteration': model.module.iteration,
                       'state_dict': model.module.Encoder.state_dict(),
                       'optimizer': model.module.optimizer.state_dict(),
                       }
            torch.save(ckpt_s2, save_model_path)

    ' Save the latest checkpoint at the end of training'
    save_model_path = os.path.join(exp_fold_dict['checkpoint'], 'model_latest.pth')
    ckpt_s2 = {'iteration': model.module.iteration,
               'state_dict': model.module.Encoder.state_dict(),
               'optimizer': model.module.optimizer.state_dict(),
               }
    torch.save(ckpt_s2, save_model_path)

    end = time.time()
    print('Iter {:d} spends {:.4f} secs'.format(config.n_iter, end - start))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    ' Load training hyperparameters from Config.py '
    config = Config()
    print('Info : ', config)

    ' Create experimental environment '
    exp_fold_dict = create_exp_env(config.exp_dir, config.exp_name)

    ' Call for multiprocessing '
    run()

    ' Add process1 to tensorboard '
    p1 = torch.multiprocessing.Process(target=tf_board_process, args=(exp_fold_dict['logger'],))
    p1.start()

    ' Add process2 to tensorboard '
    p2 = torch.multiprocessing.Process(target=train, args=(config, exp_fold_dict))
    p2.start()

    ' Start multiprocessing '
    p1.join()
    p2.join()



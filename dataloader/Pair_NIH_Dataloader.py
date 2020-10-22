import torch
from torch.utils import data
import torchvision.transforms as transforms
from .Pair_NIH_Dataset import Pair_NIH_Dataset

def Pair_NIH_Dataloader(Config, train=True):
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    #
    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #     ]),
    #     'val': transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #     ]),
    # }
    data_transforms = {
        'train': transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Grayscale(num_output_channels=3),
            # transforms.RandomRotation(45),
            # transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
    }

    datasets = {}
    datasets['train'] = Pair_NIH_Dataset(set=Config.set, seg=Config.seg, transforms=data_transforms['train'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=Config.batch_size,
                                                       shuffle=Config.shuffle, num_workers=4)

    if train:
        return dataloaders['train']
    else:
        return dataloaders['train']  # Need to be modified


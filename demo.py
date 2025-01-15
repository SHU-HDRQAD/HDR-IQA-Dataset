import os
import torch
import torchvision
import model.HDRIQA_model as models
import numpy as np
import utils.image_io as iio
import pandas as pd
from tqdm import tqdm
from scipy import stats
import argparse
from utils.algorithm import HDRPU21
import random
import cv2
from collections import OrderedDict

class QuadSplitTransform:
    def __init__(self, patch_size):
        self.size = patch_size
    def __call__(self, img):

        # get image shape
        _, width, height = img.size()
        x = width // 2 - self.size
        y = height // 2 - self.size
        a1 = random.randint(0, x)
        a2 = random.randint(0, y)
        b1 = random.randint(0, x)
        b2 = random.randint(0, y)
        c1 = random.randint(0, x)
        c2 = random.randint(0, y)
        d1 = random.randint(0, x)
        d2 = random.randint(0, y)
        # crop four pathces
        quadrants = torch.cat([
            img[:, a1:a1+self.size, a2:a2+self.size],
            img[:, width//2+b1:width // 2+b1+self.size, b2:b2+self.size],
            img[:, c1:c1+self.size, height//2+c2:height//2+c2+self.size],
            img[:, width//2+d1:width//2+d1+self.size, height//2+d2:height//2+d2+self.size],
        ],dim=0)


        return quadrants
def get_all_model_devices(model):
    devices = []
    for mdl in model.state_dict().values():
        if mdl.device not in devices:
            devices.append(mdl.device)
    return devices


def load_checkpoint(my_model, path):
    load_net = torch.load(path, map_location=get_all_model_devices(my_model)[0])
    if 'state_dict' in load_net.keys():
        load_net = load_net['state_dict']
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    my_model.load_state_dict(load_net_clean)



def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--image_path', type=str, default='./HDRImage/over_exposure_CadesCove_level_1.exr', help='Image path')
arg('--model_path', type=str, default='./parameter/pu21_for_paper_3/train_on_HDRQAD.pth', help='load model')
arg('--device', type=str, default='cuda:0', help='use cuda')
arg('--patches', type=int, default=20, help='crop patches num')
opt = parser.parse_args()

#
model_HDRIQA = models.MANIQA().cuda(opt.device)
model_HDRIQA.train(False)

load_checkpoint(model_HDRIQA, opt.model_path)
transforms = QuadSplitTransform(patch_size=224)
im_path = opt.image_path

img_ = iio.load_HDR(opt.image_path)
img_ = torch.tensor(np.transpose(img_, [2, 0, 1]).astype(np.float32))

pu21 = HDRPU21()
# random crop 20 patches and calculate mean quality score
for i in range(opt.patches):
    img = transforms(img_)
    img_pu = pu21.encode(img * 10000)
    img_pu = img_pu / (595.39 + 1e-3)
    with torch.no_grad():
        pred = model_HDRIQA(img_pu.unsqueeze(0).cuda(opt.device), img.unsqueeze(0).cuda(
            opt.device))
    pred_scores.append(float(pred.item()))
score = np.mean(pred_scores)
print('pre:%.4f' % (score))



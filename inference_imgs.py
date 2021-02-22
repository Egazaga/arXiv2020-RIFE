import os
import cv2
import torch
import argparse
from torch.nn import functional as F
from .model.RIFE_HDv2 import Model
import warnings
warnings.filterwarnings("ignore")


def inference_imgs(in_path, out_path, save_source_imgs, UHD=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    rthreshold = 0.02   # returns image when actual ratio falls in given range threshold
    rmaxcycles = 8      # limit max number of bisectional cycles

    model = Model()
    model.load_model(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train_log'), -1)
    model.eval()
    model.device()

    i = 0

    for img0_path, img1_path in zip(os.listdir(in_path), os.listdir(in_path)[1:]):
        img0 = cv2.imread(in_path + img0_path)
        img1 = cv2.imread(in_path + img1_path)
        img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

        n, c, h, w = img0.shape
        ph = ((h - 1) // 64 + 1) * 64 if UHD else ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 64 + 1) * 64 if UHD else ((w - 1) // 32 + 1) * 32

        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        out_img = model.inference(img0, img1, UHD)

        if not save_source_imgs:
            cv2.imwrite(f'{out_path}/{str(i).zfill(6)}.png',
                        (out_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
            i += 1
        else:
            cv2.imwrite(f'{out_path}/{str(i).zfill(6)}.png',
                        (img0[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
            i += 1
            cv2.imwrite(f'{out_path}/{str(i).zfill(6)}.png',
                        (out_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
            i += 1
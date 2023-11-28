import os
import numpy as np
import argparse
from PIL import Image
from myutils.quality_no_refer import calculate_path_NRIQA
from myutils.quality_refer import calc_psnr, calc_mse, calc_ssim

parser = argparse.ArgumentParser(description='Evaluating UIEB dataset')

parser.add_argument('--method_name', type=str, default='UIEC2Net', 
                    help='method name, any subfolder in ./data/UIEB/All_Results/')
parser.add_argument('--folder', type=str, default='T90', choices=['T90', 'C60'],
                    help='options:[T90 or C60]')

hparams = parser.parse_args()

if hparams.folder == 'T90':
    gt_path = "./data/UIEB/reference-890/"   # GT path
else:
    gt_path = None

test_path = "./data/UIEB/All_Results/" + hparams.method_name + "/" + hparams.folder + "/"  # the pred images

PSNR_list = []
SSIM_list = []
MSE_list = []

if gt_path is not None:
    for filename in os.listdir(test_path):
        file_path = os.path.join(test_path, filename)
        file_path_gt = os.path.join(gt_path, filename)
        pred_img = Image.open(file_path)
        gt_img = Image.open(file_path_gt)
        gt_img = gt_img.resize((pred_img.size))
        pred_img = np.array(pred_img) / 255.
        gt_img = np.array(gt_img) / 255.
        PSNR_list.append(calc_psnr(pred_img, gt_img, is_for_torch=False))
        SSIM_list.append(calc_ssim(pred_img, gt_img, is_for_torch=False))
        MSE_list.append(calc_mse(pred_img, gt_img, is_for_torch=False).item())

    print("PSNR: ", np.mean(PSNR_list))
    print("SSIM: ", np.mean(SSIM_list))
    print("MSE: ", np.mean(MSE_list))

calculate_path_NRIQA(test_path)


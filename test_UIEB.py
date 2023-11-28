import os
import torch
import argparse
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.UIEB import UIEBDataset
from model.UWCNN import UWCNN
from model.UIEC2Net import UIEC2Net
from model.NU2Net import NU2Net
from model.FIVE_APLUS import FIVE_APLUSNet
from model.UTrans import UTrans

from myutils.quality_refer import calc_psnr, calc_mse, calc_ssim, normalize_img


parser = argparse.ArgumentParser(description='Testing UIEB dataset')

parser.add_argument('--model_name', type=str, default='UIEC2Net', 
                    help='model name, options:[UIEC2Net, UTrans, NU2Net, UWCNN, FIVE_APLUS]')
parser.add_argument('--crop_size', type=int, default=256, help='crop size')
parser.add_argument('--input_norm', action='store_true', help='norm the input image to [-1,1]')

hparams = parser.parse_args()

model_path = './checkpoints/UIEB/' + hparams.model_name + '.ckpt'

test_path = './data/UIEB/All_Results/' + hparams.model_name + '/T90/'
pred_path = './data/UIEB/All_Results/' + hparams.model_name + '/C60/'
if not os.path.exists(test_path):
    os.makedirs(test_path)
if not os.path.exists(pred_path):
    os.makedirs(pred_path)

test_set = UIEBDataset("./data/", train_flag=False, pred_flag=False, train_size=hparams.crop_size, input_norm=hparams.input_norm)  # T90
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

pred_set = UIEBDataset("./data/", train_flag=False, pred_flag=True, train_size=hparams.crop_size, input_norm=hparams.input_norm)  # C60
pred_loader = DataLoader(pred_set, batch_size=1, shuffle=False)

model_zoos = {
    "UWCNN": UWCNN,
    "UIEC2Net": UIEC2Net,
    "NU2Net": NU2Net,
    "FIVE_APLUS": FIVE_APLUSNet,
    "UTrans": UTrans,
    }
model = model_zoos[hparams.model_name]().cuda()
ckpt = torch.load(model_path)
ckpt = ckpt['state_dict']
new_ckpt =  {key[6:]: value for key, value in ckpt.items()}
missing_keys, unexpected_keys = model.load_state_dict(new_ckpt, strict=False)
print("missing keys: ", missing_keys)
print("unexpected keys: ", unexpected_keys)
model.eval()

print("generate enhanced images for test set (90 images)")
for idx, (x, y, filename) in tqdm(enumerate(test_loader),total=len(test_loader)):
    with torch.no_grad():
        x = x.cuda()
        y_hat = model(x)
        gt_img = y[0].permute(1,2,0).detach().cpu().numpy()

        pred_img_tensor = normalize_img(y_hat)
        pred_img = pred_img_tensor[0].permute(1,2,0).detach().cpu().numpy()

        save_image(pred_img_tensor[0], os.path.join(test_path, filename[0]), normalize=False)

print("generate enhanced images for challenging set (60 images)")
for idx, (x, y, filename) in tqdm(enumerate(pred_loader),total=len(pred_loader)):
    with torch.no_grad():
        x = x.cuda()
        y_hat = model(x)
        pred_img_tensor = normalize_img(y_hat)
        pred_img = pred_img_tensor[0].permute(1,2,0).detach().cpu().numpy()

        save_image(pred_img_tensor[0], os.path.join(pred_path, filename[0]), normalize=False)



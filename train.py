import os
import wandb
import numpy as np
from argparse import Namespace

import pyiqa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

# from archs.FIVE_APLUS import FIVE_APLUSNet
from model.UWCNN import UWCNN
from model.UIEC2Net import UIEC2Net
from model.NU2Net import NU2Net
from model.FIVE_APLUS import FIVE_APLUSNet
from model.UTrans import UTrans
from dataset.UIEB import UIEBDataset
from dataset.LSUI import LSUIDataset
from myutils.losses import *
from myutils.quality_refer import calc_psnr, calc_mse, calc_ssim, normalize_img


class TrainUIEModel(pl.LightningModule):
    def __init__(self, hparams):
        super(TrainUIEModel, self).__init__()

        model_zoos = {
            "UWCNN": UWCNN,
            "UIEC2Net": UIEC2Net,
            "NU2Net": NU2Net,
            "FIVE_APLUS": FIVE_APLUSNet,
            "UTrans": UTrans,
            }

        self.params = hparams

        # Train setting
        self.initlr = self.params.initlr  # initial learning
        self.weight_decay = self.params.weight_decay  # optimizers weight decay
        self.lr_config = self.params.lr_config

        self.ssim_loss = pyiqa.create_metric('ssim', as_loss=True, device=self.device)

        vgg_model = vgg16(weights = VGG16_Weights.IMAGENET1K_V1).cuda().eval()
        vgg_model = vgg_model.features[:16]
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.per_loss = PerpetualLoss(vgg_model=vgg_model)

        self.l1_loss = MyLoss()
        self.char_loss = CharLoss()

        self.val_ssim = pyiqa.create_metric('ssim', device=self.device)
        self.val_psnr = pyiqa.create_metric('psnr', device=self.device)

        self.model = model_zoos[hparams.model_name]()
        self.save_hyperparameters()

    def forward(self, x):
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initlr, betas=[0.9,0.999], weight_decay=self.weight_decay)
        if self.lr_config == 'CyclicLR':
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.initlr,
                max_lr=1.2*self.initlr,
                cycle_momentum=False
                )
        elif self.lr_config == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.8
                )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = self.l1_loss(y_hat, y) + 0.2*self.per_loss(y_hat, y)  # - 0.5*self.ssim_loss(y_hat, y)

        self.log('train_loss', loss, sync_dist=True, batch_size=x.shape[0])
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        assert x.shape[0] == 1
        y_hat = self.forward(x)
        
        _, _, h, w = y.shape
        gt_img = y[0].permute(1,2,0).detach().cpu().numpy()

        upsample = nn.UpsamplingBilinear2d((h, w))
        pred_img = upsample(normalize_img(y_hat))
        pred_img = pred_img[0].permute(1,2,0).detach().cpu().numpy()

        psnr = calc_psnr(pred_img, gt_img, is_for_torch=False)
        ssim = calc_ssim(pred_img, gt_img, is_for_torch=False)
        mse = calc_mse(pred_img, gt_img, is_for_torch=False)

        self.log('psnr', psnr, sync_dist=True, batch_size=1)
        self.log('ssim', ssim, sync_dist=True, batch_size=1)
        self.log('mse', mse, sync_dist=True, batch_size=1)

        if batch_idx==0:
            self.logger.experiment.log({
                    "raw": [wandb.Image(x[0], caption="raw")],
                    "gt": [wandb.Image(gt_img, caption="gt")],
                    "pred": [wandb.Image(pred_img, caption="pred")]
                    })

        return {'psnr': psnr, 'ssim': ssim, 'mse': mse}


def main():
    args = {
        'crop_size':256,
        'input_norm':False,
        'epochs':100,
        'batch_size':8,
        'num_workers':4,
        'initlr':0.001,
        'weight_decay':0.000,
        'model_name':"UTrans",
        'lr_config':'CyclicLR',
    }
    # UTrans
    # args = {
    #     'crop_size':256,
    #     'input_norm':False,
    #     'epochs':150,
    #     'batch_size':8,
    #     'num_workers':4,
    #     'initlr':0.0005,
    #     'weight_decay':0.001,
    #     'model_name':"UTrans",
    #     'lr_config':'StepLR',
    # }

    hparams = Namespace(**args)

    seed = 42
    seed_everything(seed)

    logger = WandbLogger(
        project="UIE",
        name=hparams.model_name,
        log_model=True
        )
    
    RESUME = False
    checkpoint_path = "./checkpoints/UIEB/"

    train_set = UIEBDataset("./data/", train_flag=True, pred_flag=False, train_size=hparams.crop_size, input_norm=hparams.input_norm)
    test_set = UIEBDataset("./data/", train_flag=False, pred_flag=False, train_size=hparams.crop_size, input_norm=hparams.input_norm)
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=hparams.batch_size, 
        shuffle=True, 
        num_workers=hparams.num_workers
        )
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=1, 
        shuffle=False,
        num_workers=hparams.num_workers
        )
   
    model = TrainUIEModel(hparams)

    checkpoint_callback = ModelCheckpoint(
        monitor='psnr',
        dirpath=checkpoint_path,
        filename='epoch{epoch:02d}-psnr{psnr:.3f}-ssim{ssim:.3f}',
        auto_insert_metric_name=False,
        every_n_epochs=1,
        save_top_k=3,
        mode="max",
        save_last=True,
        save_weights_only=True
    )

    if RESUME:
        trainer = pl.Trainer(
            max_epochs=hparams.epochs,
            resume_from_checkpoint = checkpoint_path,
            devices=[0],
            logger=logger,
            accelerator='cuda',
            callbacks=[checkpoint_callback],
            gradient_clip_val=0.5, 
            gradient_clip_algorithm="value",
        ) 
    else:
        trainer = pl.Trainer(
            max_epochs=hparams.epochs,
            devices=[0],
            logger=logger,
            accelerator='cuda',
            callbacks=[checkpoint_callback],
            gradient_clip_val=0.5,
            gradient_clip_algorithm="value",
            log_every_n_steps=5,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0
        )

    trainer.fit(model, train_loader, test_loader)
    

if __name__ == '__main__':
    main()






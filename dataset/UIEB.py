import os
import random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF


class UIEBDataset(data.Dataset):
    def __init__(self, data_path, train_flag=True, pred_flag=False, train_size=256, input_norm=False):
        super(UIEBDataset, self).__init__()
        self.data_path = data_path
        self.train_flag = train_flag if not pred_flag else False
        self.train_size = train_size
        self.pred_flag = pred_flag
        self.input_norm = input_norm
        if self.train_flag:
            self.ann_file = os.path.join(self.data_path, "UIEB", "train.txt")
        else:
            self.ann_file = os.path.join(self.data_path, "UIEB", "test.txt")
        if self.pred_flag:
            self.ann_file = os.path.join(self.data_path, "UIEB", "challenging.txt")
            self.data_infos = self.load_unpaired()
        else:
            self.data_infos = self.load_annotations()

    def load_annotations(self):
        data_infos = []
        with open(self.ann_file, 'r') as f:
            data_list = f.read().splitlines()
            for data in data_list:
                data_infos.append({
                    "image_path": os.path.join(self.data_path, "UIEB", "raw-890", data),
                    "gt_path": os.path.join(self.data_path, "UIEB", "reference-890", data),
                    "filename": data,
                })
        return data_infos

    def load_unpaired(self):
        data_infos = []
        with open(self.ann_file, 'r') as f:
            data_list = f.read().splitlines()
            for data in data_list:
                data_infos.append({
                    "image_path": os.path.join(self.data_path, "UIEB", "challenging-60", data),
                    "filename": data,
                })
        return data_infos

    def augData(self, data, target):
        if not self.pred_flag:
            if self.train_flag:  # train
                data = tfs.Resize([self.train_size, self.train_size])(data)
                target = tfs.Resize([self.train_size, self.train_size])(target)
                rand_hor = random.randint(0, 1)
                rand_rot = random.randint(0, 3)
                data = tfs.RandomHorizontalFlip(rand_hor)(data)
                target = tfs.RandomHorizontalFlip(rand_hor)(target)
                if rand_rot:
                    data = FF.rotate(data,90*rand_rot)
                    target = FF.rotate(target,90*rand_rot)
            else:                # test
                data = tfs.Resize([self.train_size, self.train_size])(data)
                target = tfs.Resize([self.train_size, self.train_size])(target)
            data = tfs.ToTensor()(data)
            target = tfs.ToTensor()(target)
            if self.input_norm:
                data = tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(data)
                target = tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(target)
        else:                    # pred
            data = tfs.Resize([self.train_size, self.train_size])(data)
            data = tfs.ToTensor()(data)
            target = tfs.ToTensor()(target)
            if self.input_norm:
                data = tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(data)

        return data, target

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        result = self.data_infos[idx]
        data = Image.open(result['image_path']).convert('RGB')
        if not self.pred_flag:
            target = Image.open(result['gt_path']).convert('RGB')
        else:
            target = Image.open(result['image_path']).convert('RGB')
        data, target = self.augData(data, target)
        return data, target, result["filename"]

import os
import random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF


class U45Dataset(data.Dataset):
    def __init__(self, data_path, train_size=256, input_norm=False):
        super(U45Dataset, self).__init__()
        self.data_path = data_path
        self.train_size = train_size
        self.input_norm = input_norm
        self.data_infos = self.load_unpaired()

    def load_unpaired(self):
        data_infos = []
        for data in os.listdir(self.data_path):
            data_infos.append({
                "image_path": os.path.join(self.data_path, data),
                "filename": data,
            })
        return data_infos

    def augData(self, data, target):
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
        target = Image.open(result['image_path']).convert('RGB')
        data, target = self.augData(data, target)
        return data, target, result["filename"]

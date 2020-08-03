from torch.utils.data import Dataset, DataLoader
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1, 3"
import torch
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms



class Pix2PixDataset(Dataset):
    def __init__(self, is_train):
        base_path = os.path.join('/media/data2/dataset/Pix2pix/maps', 'train' if is_train else 'val')
        self.total_data = glob.glob(os.path.join(base_path, "*.*"))
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.total_data)

    def __getitem__(self, index):
        pair_image = cv2.imread(self.total_data[index])
        data_x = self.transform(pair_image[:, :pair_image.shape[1] // 2, :])
        data_y = self.transform(pair_image[:, pair_image.shape[1] // 2:, :])
        return {'data_x': data_x, 'data_y': data_y}

if __name__ == "__main__":
    dataset = Pix2PixDataset(True)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True, pin_memory=True, num_workers=2)
    for idx, datas in enumerate(data_loader):
        print(type(datas))
        print(torch.mean(datas['data_x']))
        print(torch.max(datas['data_x']))
        print(torch.min(datas['data_x']))
        # plt.imshow(datas['data_x'][0].cpu().detach().numpy())
        # plt.show()
        exit()

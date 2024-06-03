import numpy as np
import imageio
import os
import torch
from torch.utils.data import Dataset, DataLoader

class CASIA_Face(Dataset):
    def __init__(self, root):
        self.root = root

        img_txt_dir = os.path.join(root, 'CASIA-WebFace-112X96.txt')
        image_list = []
        label_list = []
        with open(img_txt_dir) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:
            image_dir, label_name = info.split(' ')
            image_list.append(os.path.join(root, 'CASIA-WebFace-112X96', image_dir))
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))

    def __getitem__(self, index):
        img_path = self.image_list[index]
        target = self.label_list[index]
        img = imageio.imread(img_path)

        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=2)
        flip = np.random.choice(2)*2-1
        img = img[:, ::flip, :]
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)

        # Create the imaginary part as random noise
        img_real = img
        img_imag = np.random.normal(0, 1, img.shape).astype(np.float32)

        # Stack the real and imaginary parts along a new dimension
        img_complex = np.stack([img_real, img_imag], axis=0)

        img_complex = torch.from_numpy(img_complex).float()

        return img_complex, target

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    data_dir = '/home/brl/USER/fzc/dataset/CASIA'
    dataset = CASIA_Face(root=data_dir)
    trainloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=False)
    print(len(dataset))
    for data in trainloader:
        print(data[0].shape)

import os
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset

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

        if len(img.shape) == 2:  # If image is grayscale
            img = np.expand_dims(img, axis=-1)  # Add one channel dimension

        flip = np.random.choice(2)*2-1
        img = img[:, ::flip, :]
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)

        # Create the imaginary part as zeros
        img_real = img
        img_imag = np.zeros_like(img)

        # Stack the real and imaginary parts along a new dimension
        img_complex = np.stack([img_real, img_imag], axis=0)

        img_complex = torch.from_numpy(img_complex).float()

        return img_complex, target

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    pass

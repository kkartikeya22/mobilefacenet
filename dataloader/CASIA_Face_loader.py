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

        # Convert RGB to grayscale
        if len(img.shape) == 3:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

        # Add channel dimension
        img = np.expand_dims(img, axis=-1)

        # Ensure the number of channels is 2
        img = np.repeat(img, 2, axis=-1)

        # Normalize image
        img = (img - 127.5) / 128.0

        # Create complex input (real and imaginary parts)
        img_real = img
        img_imag = np.zeros_like(img)
        img_complex = np.stack([img_real, img_imag], axis=0)

        img_complex = torch.from_numpy(img_complex).float()

        return img_complex, target

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    pass

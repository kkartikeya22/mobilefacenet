import numpy as np
import imageio
import torch
from torch.utils.data import Dataset

class LFW(Dataset):
    def __init__(self, imgl, imgr):
        self.imgl_list = imgl
        self.imgr_list = imgr

    def __getitem__(self, index):
        imgl = imageio.imread(self.imgl_list[index])
        if len(imgl.shape) == 2:
            imgl = np.stack([imgl] * 3, axis=2)
        imgr = imageio.imread(self.imgr_list[index])
        if len(imgr.shape) == 2:
            imgr = np.stack([imgr] * 3, axis=2)

        imglist = [imgl, imgl[:, ::-1, :], imgr, imgr[:, ::-1, :]]
        for i in range(len(imglist)):
            imglist[i] = (imglist[i] - 127.5) / 128.0
            imglist[i] = imglist[i].transpose(2, 0, 1)
        
        # Convert to complex data
        complex_imglist = []
        for img in imglist:
            img_real = img
            img_imag = np.random.normal(0, 1, img.shape).astype(np.float32)
            img_complex = np.stack([img_real, img_imag], axis=0)
            complex_imglist.append(torch.from_numpy(img_complex).float())

        return complex_imglist

    def __len__(self):
        return len(self.imgl_list)

if __name__ == '__main__':
    pass

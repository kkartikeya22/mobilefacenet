import os
import torch
import torch.optim as optim
import torch.utils.data
from torch.optim import lr_scheduler
from datetime import datetime
from config import BATCH_SIZE, SAVE_FREQ, RESUME, SAVE_DIR, TEST_FREQ, TOTAL_EPOCH, MODEL_PRE, GPU
from config import CASIA_DATA_DIR, LFW_DATA_DIR
from core.model import ComplexMobileFacenet
from dataloader.CASIA_Face_loader import CASIA_Face
from dataloader.LFW_loader import LFW
from lfw_eval import parseList, evaluation_10_fold
import time
import numpy as np
import scipy.io
import logging

# Initialize logging
def init_log(save_dir):
    log_filename = os.path.join(save_dir, 'train.log')
    level = logging.INFO
    format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=log_filename, level=level, format=format)
    console = logging.StreamHandler()
    console.setLevel(level)
    formatter = logging.Formatter(format)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info("Logging initialized.")

# GPU initialization
gpu_list = ''
multi_gpus = False
if isinstance(GPU, int):
    gpu_list = str(GPU)
else:
    multi_gpus = True
    for i, gpu_id in enumerate(GPU):
        gpu_list += str(gpu_id)
        if i != len(GPU) - 1:
            gpu_list += ','
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

# Other initialization
start_epoch = 1
save_dir = os.path.join(SAVE_DIR, MODEL_PRE + 'v2_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)
logging = init_log(save_dir)
_print = logging.info

# Define trainloader and testloader
trainset = CASIA_Face(root=CASIA_DATA_DIR)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2, drop_last=False)

nl, nr, folds, flags = parseList(root=LFW_DATA_DIR)
testdataset = LFW(nl, nr)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=32,
                                         shuffle=False, num_workers=2, drop_last=False)

# Define model
net = ComplexMobileFacenet()

if RESUME:
    ckpt = torch.load(RESUME)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1

# Define optimizer and scheduler
optimizer_ft = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=4e-5)
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[36, 52, 58], gamma=0.1)

net = net.cuda()
if multi_gpus:
    net = torch.nn.DataParallel(net)
criterion = torch.nn.CrossEntropyLoss()

best_acc = 0.0
best_epoch = 0
for epoch in range(start_epoch, TOTAL_EPOCH + 1):
    exp_lr_scheduler.step()
    # Training
    _print('Train Epoch:{}_{} ...'.format(epoch, TOTAL_EPOCH))
    net.train()

    train_total_loss = 0.0
    total = 0
    since = time.time()
    for data in trainloader:
        img, label = data[0].cuda(), data[1].cuda()
        batch_size = img.size(0)
        optimizer_ft.zero_grad()

        output = net(img)

        total_loss = criterion(output, label)
        total_loss.backward()
        optimizer_ft.step()

        train_total_loss += total_loss.item() * batch_size
        total += batch_size

    train_total_loss = train_total_loss / total
    time_elapsed = time.time() - since
    loss_msg = '    total_loss: {:.4f} time: {:.0f}m {:.0f}s'.format(train_total_loss, time_elapsed // 60, time_elapsed % 60)
    _print(loss_msg)

    # Test on LFW
    if epoch % TEST_FREQ == 0:
        net.eval()
        featureLs = None
        featureRs = None
        _print('Test Epoch: {} ...'.format(epoch))
        for data in testloader:
            for i in range(len(data)):
                data[i] = data[i].cuda()
            res = [net(d).data.cpu().numpy() for d in data]
            featureL = np.concatenate((res[0], res[1]), 1)
            featureR = np.concatenate((res[2], res[3]), 1)
            if featureLs is None:
                featureLs = featureL
            else:
                featureLs = np.concatenate((featureLs, featureL), 0)
            if featureRs is None:
                featureRs = featureR
            else:
                featureRs = np.concatenate((featureRs, featureR), 0)

        result = {'fl': featureLs, 'fr': featureRs, 'fold': folds, 'flag': flags}
        scipy.io.savemat(os.path.join(save_dir, 'epoch_{}.mat'.format(epoch)), result)

        accs = evaluation_10_fold(result)
        _print('LFW Ave Accuracy: {:.4f}'.format(np.mean(accs) * 100))

        if np.mean(accs) > best_acc:
            best_acc = np.mean(accs)
            best_epoch = epoch
            _print('Best LFW Ave Accuracy: {:.4f}, Achieved at epoch {}'.format(best_acc * 100, best_epoch))
            if multi_gpus:
                torch.save({'epoch': epoch, 'net_state_dict': net.module.state_dict()},
                           os.path.join(save_dir, 'best_model.pth'))
            else:
                torch.save({'epoch': epoch, 'net_state_dict': net.state_dict()},
                           os.path.join(save_dir, 'best_model.pth'))

    # Save model
    if epoch % SAVE_FREQ == 0:
        if multi_gpus:
            torch.save({'epoch': epoch, 'net_state_dict': net.module.state_dict()},
                       os.path.join(save_dir, 'model_{}.pth'.format(epoch)))
        else:
            torch.save({'epoch': epoch, 'net_state_dict': net.state_dict()},
                       os.path.join(save_dir, 'model_{}.pth'.format(epoch)))

_print('Finished Training')

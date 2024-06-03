import os
import torch.utils.data
from torch import nn
from torch.nn import DataParallel
from datetime import datetime
from config import BATCH_SIZE, SAVE_FREQ, RESUME, SAVE_DIR, TEST_FREQ, TOTAL_EPOCH, MODEL_PRE, GPU
from config import CASIA_DATA_DIR, LFW_DATA_DIR
from core import model
from core.utils import init_log
from dataloader.CASIA_Face_loader import CASIA_Face
from dataloader.LFW_loader import LFW
from torch.optim import lr_scheduler
import torch.optim as optim
import time
from lfw_eval import parseList, evaluation_10_fold
import numpy as np
import scipy.io

# GPU initialization
gpu_list = ','.join(map(str, GPU)) if isinstance(GPU, list) else str(GPU)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

# Directory initialization
start_epoch = 1
save_dir = os.path.join(SAVE_DIR, MODEL_PRE + 'v2_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)
logging = init_log(save_dir)
_print = logging.info

# Data loaders
trainset = CASIA_Face(root=CASIA_DATA_DIR)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2, drop_last=False)

nl, nr, folds, flags = parseList(root=LFW_DATA_DIR)
testdataset = LFW(nl, nr)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=32,
                                         shuffle=False, num_workers=2, drop_last=False)

# Model definition
net = model.MobileFacenet()
ArcMargin = model.ArcMarginProduct(128, trainset.class_nums)

if RESUME:
    ckpt = torch.load(RESUME)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1

# Optimizers
ignored_params = list(map(id, net.linear1.parameters())) + list(map(id, ArcMargin.weight))
prelu_params = [p for m in net.modules() if isinstance(m, nn.PReLU) for p in m.parameters()]
ignored_params += list(map(id, prelu_params))

base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

optimizer_ft = optim.SGD([
    {'params': base_params, 'weight_decay': 4e-5},
    {'params': net.linear1.parameters(), 'weight_decay': 4e-4},
    {'params': ArcMargin.weight, 'weight_decay': 4e-4},
    {'params': prelu_params, 'weight_decay': 0.0}
], lr=0.1, momentum=0.9, nesterov=True)

exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[36, 52, 58], gamma=0.1)

net = net.cuda()
ArcMargin = ArcMargin.cuda()
if isinstance(GPU, list):
    net = DataParallel(net)
    ArcMargin = DataParallel(ArcMargin)
criterion = nn.CrossEntropyLoss()

best_acc = 0.0
best_epoch = 0
for epoch in range(start_epoch, TOTAL_EPOCH+1):
    exp_lr_scheduler.step()
    _print(f'Train Epoch: {epoch}/{TOTAL_EPOCH} ...')
    net.train()

    train_total_loss = 0.0
    total = 0
    since = time.time()
    for data in trainloader:
        img, label = data[0].cuda(), data[1].cuda()
        batch_size = img.size(0)
        optimizer_ft.zero_grad()

        raw_logits = net(img)
        output = ArcMargin(raw_logits, label)
        total_loss = criterion(output, label)
        total_loss.backward()
        optimizer_ft.step()

        train_total_loss += total_loss.item() * batch_size
        total += batch_size

    train_total_loss /= total
    time_elapsed = time.time() - since
    _print(f'    total_loss: {train_total_loss:.4f} time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # Test model on LFW
    if epoch % TEST_FREQ == 0:
        net.eval()
        featureLs = []
        featureRs = []
        _print(f'Test Epoch: {epoch} ...')
        with torch.no_grad():
            for data in testloader:
                data = [d.cuda() for d in data]
                res = [net(d).data.cpu().numpy() for d in data]
                featureL = np.concatenate((res[0], res[1]), 1)
                featureR = np.concatenate((res[2], res[3]), 1)
                featureLs.append(featureL)
                featureRs.append(featureR)

        featureLs = np.vstack(featureLs)
        featureRs = np.vstack(featureRs)

        result = {'fl': featureLs, 'fr': featureRs, 'fold': folds, 'flag': flags}
        scipy.io.savemat('./result/tmp_result.mat', result)
        accs = evaluation_10_fold('./result/tmp_result.mat')
        _print(f'    ave: {np.mean(accs) * 100:.4f}')

    # Save model
    if epoch % SAVE_FREQ == 0:
        _print(f'Saving checkpoint: {epoch}')
        net_state_dict = net.module.state_dict() if isinstance(GPU, list) else net.state_dict()
        torch.save({'epoch': epoch, 'net_state_dict': net_state_dict},
                   os.path.join(save_dir, f'{epoch:03d}.ckpt'))

_print('Finished training')

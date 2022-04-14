import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import NEmon
import numpy as np
import time
from logger import Logger
import numpy as np
from tqdm import tqdm
import wandb

def cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def pretrain(trainLoader, testLoader, model, epochs=15, max_lr=1e-3, change_mo=True,  lr_mode='step',
          step=10):

    optimizer = optim.Adam(model.parameters(), lr=max_lr)
    lr_scheduler =optim.lr_scheduler.StepLR(optimizer, step, gamma=0.1, last_epoch=-1)


    model = cuda(ExplicitNet(model))

    for epoch in range(1, 1 + epochs):
        nProcessed = 0
        nTrain = len(trainLoader.dataset)
        model.train()
        start = time.time()
        total_train_loss = 0
        for batch_idx, batch in enumerate(trainLoader):

            data, target = cuda(batch[0]), cuda(batch[1])
            optimizer.zero_grad()
            preds = model(data)
            ce_loss = F.cross_entropy(preds, target)

            ce_loss.backward()

            total_train_loss += ce_loss.item()
            nProcessed += len(data)

            optimizer.step()
            

        

        if lr_mode == 'step':
            lr_scheduler.step()

        print("Tot pretrain time: {}".format(time.time() - start))

        start = time.time()
        test_loss = 0
        incorrect = 0
        model.eval()
        with torch.no_grad():
            for batch in testLoader:
                data, target = cuda(batch[0]), cuda(batch[1])
                preds = model(data)
                ce_loss = nn.CrossEntropyLoss(reduction='sum')(preds, target)
                test_loss += ce_loss
                incorrect += preds.float().argmax(1).ne(target.data).sum()
            test_loss /= len(testLoader.dataset)
            nTotal = len(testLoader.dataset)
            err = 100. * incorrect.float() / float(nTotal)

            print('\n\nPre-train set: Average loss: {:.4f}, Error: {}/{} ({:.2f}%)'.format(
                test_loss, incorrect, nTotal, err))

                    
        print("Tot test time: {}\n\n\n\n".format(time.time() - start))




class ExplicitNet(nn.Module):

    def __init__(self, im_model, in_dim=784, out_dim=100, m=0.1, **kwargs):
        super().__init__()
        self.linear_module = im_model.mon.linear_module
        self.nonlin_module = im_model.mon.nonlin_module
        self.Wout = im_model.Wout


        self.iters = 6
        self.last_z = None
        #self.D = nn.Linear(in_dim, 10, bias=False)

    def forward(self, x):
        # x = x.view(x.shape[0], -1)
        # if self.last_z is None:
        #     self.last_z = tuple(torch.zeros(s, dtype=x.dtype, device=x.device)
        #               for s in self.linear_module.z_shape(x.shape[0]))
        # z = self.last_z

        # for i in range(self.iters):
        #     zn = self.nonlin_module(*self.linear_module(x, *z))
        #     z = zn
        # self.last_z = zn
        # return self.Wout(zn[-1])# + self.D(x)


        x = x.view(x.shape[0], -1)
        
        last_z = tuple(torch.zeros(s, dtype=x.dtype, device=x.device)
                      for s in self.linear_module.z_shape(x.shape[0]))

        for i in range(self.iters):
            zn = self.nonlin_module(*self.linear_module(x, *last_z))
            last_z = zn
        # Two options: for loop, and dist reg.

        return self.Wout(zn[-1])# + self.D(x)
        





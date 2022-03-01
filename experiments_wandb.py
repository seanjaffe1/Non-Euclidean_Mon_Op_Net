"""
Experiments
"""

import splitting as sp
import train
import torchvision
import torchvision.transforms as transforms
import torch
import argparse
import numpy as np
import wandb




parser = argparse.ArgumentParser()
parser.add_argument('--dataset', metavar='name', nargs=1, default='mnist', 
                    help="Which dataset to use: 'cifar' or 'mnist'")
parser.add_argument('--epochs', type=int, default=10,
                    help='training epochs')
parser.add_argument('--batch_size', type=int, default=120,
                    help='size of training batch')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--wandb', action='store_true',
                    help='Log run in wandb')
parser.add_argument('--notes', type=ascii, default='',
                    help='tag string to add to wandb log')

parser.add_argument('--detect_anomaly', action='store_true',
                    help='sets torch.autograd,set_detect_anomaly to True')
parser.add_argument('--seed', type=int, default=42,
                    help='seed for numpy and pytorch')

args = parser.parse_args()

dataset = args.dataset
epochs = args.epochs
batch_size=args.batch_size
learning_rate =args.lr
log_wandb=args.wandb

seed_no=args.seed
torch.manual_seed(seed_no)
np.random.seed(seed_no)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_no)
    torch.cuda.empty_cache()

if args.detect_anomaly:
     torch.autograd.set_detect_anomaly(True)

if log_wandb:
     print("Logging in wandb")
     wandb.init(project="Nemon", notes=args.notes, group=dataset, entity="seanjaffe1")


     wandb.config.learning_rate = learning_rate,
     wandb.config.batch_size = batch_size,
     wandb.config.epochs = epochs
     wandb.config.seed_no = seed_no



if dataset == 'cifar':
     trainLoader, testLoader = train.cifar_loaders(train_batch_size=batch_size, test_batch_size=200, augment=False)
elif dataset == 'mnist':
     trainLoader, testLoader = train.mnist_loaders(train_batch_size=batch_size, test_batch_size=200)

if dataset == 'cifar':
     train.train(trainLoader, testLoader,
               train.NESingleConvNet(sp.NEmonForwardStep,
                                   in_dim=32,
                         in_channels=3,
                         out_channels=81,
                         alpha=0.5,
                         max_iter=300,
                         tol=1e-2,
                         m=0),
          max_lr=learning_rate,
          lr_mode='step',
          step=25,
          change_mo=False,
          epochs=epochs,
          print_freq=100,
          tune_alpha=False,
          regularizer=0,
          log_wandb=log_wandb)

if dataset == 'mnist':
     # train.train(trainLoader, testLoader,
     #      train.NESingleFcNet(sp.NEmonForwardStep,
     #                     in_dim=28**2,
     #                     out_dim=100,
     #                     alpha=1.0,
     #                     max_iter=300,
     #                     tol=1e-2,
     #                     m=0.0),
     #      max_lr=learning_rate,
     #      lr_mode='step',
     #      step=10,
     #      change_mo=False,
     # #       epochs=40,
     #      epochs=epochs,
     #      print_freq=100,
     #      tune_alpha=False,
     #      regularizer = 0,
     #      log_wandb=log_wandb)

     train.train(trainLoader, testLoader,
                train.NESingleConvNet(sp.NEmonForwardStep,
                                     in_dim=28,
                            in_channels=1,
                            out_channels=64,
                            alpha=0.5,
                            max_iter=300,
                            tol=1e-2,
                            m=0),
            max_lr=learning_rate,
            lr_mode='step',
            step=25,
            change_mo=False,
            epochs=epochs,
            print_freq=100,
            tune_alpha=False,
            regularizer=0,
            log_wandb=log_wandb)
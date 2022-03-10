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
parser.add_argument('--dataset', metavar='name', default='mnist', 
                    help="Which dataset to use: 'cifar' or 'mnist'")
parser.add_argument('--model', metavar='name', default='scnn', 
                    help="Which model to use: 'fnn', 'scnn', 'mcnn'")

parser.add_argument('--epochs', type=int, default=10,
                    help='training epochs')
parser.add_argument('--batch_size', type=int, default=200,
                    help='size of training batch')
parser.add_argument('--test_batch_size', type=int, default=250,
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
parser.add_argument('--m', type=float, default=1.0)

parser.add_argument('--sp', metavar='name', default='ForwardStep',
                    help = 'Splitting method: \'FS\'  for ForwardStep or \'PR\' for PeacemanRachford' )
args = parser.parse_args()

dataset = args.dataset
model_type = args.model
epochs = args.epochs
batch_size=args.batch_size
test_batch_size=args.test_batch_size
learning_rate =args.lr
log_wandb=args.wandb
m = args.m
sp = args.sp
 
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

     wandb.config.model = model_type
     wandb.config.learning_rate = learning_rate,
     wandb.config.batch_size = batch_size,
     wandb.config.epochs = epochs
     wandb.config.seed_no = seed_no
     wandb.config.m = m
     wandb.config.sp = sp


if dataset == 'cifar':
     trainLoader, testLoader = train.cifar_loaders(train_batch_size=batch_size, test_batch_size=batch_size, augment=False)
elif dataset == 'mnist':
     trainLoader, testLoader = train.mnist_loaders(train_batch_size=test_batch_size, test_batch_size=batch_size)

if dataset == 'cifar':
     if model_type == 'scnn':
          model = train.NESingleConvNet(sp.NEmonForwardStep,
                                   in_dim=32,
                         in_channels=3,
                         out_channels=81,
                         alpha=0.5,
                         max_iter=300,
                         tol=1e-2,
                         m=m)
     else:
          raise argparse.ArgumentError("model must be 'scnn'")    

     train.train(trainLoader, testLoader,
          model,
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
     if model_type == 'fnn':
          model = train.NESingleFcNet(sp.NEmonForwardStep,
                         in_dim=28**2,
                         out_dim=100,
                         alpha=1.0,
                         max_iter=300,
                         tol=1e-2,
                         m=m)
          step = 10
     elif model_type == 'scnn':

          model = train.NESingleConvNet(sp.NEmonForwardStep,
                                in_dim=28,
                       in_channels=1,
                       out_channels=64,
                       alpha=0.5,
                       max_iter=300,
                       tol=1e-2,
                       m=m)
          step  = 25

     elif model_type == 'mcnn':

          model = train.NEMultiConvNet(sp.NEmonForwardStep,
                                in_dim=28,
                       conv_sizes=(16, 32, 64),
                       alpha=0.5,
                       max_iter=300,
                       tol=1e-2,
                       m=m)
          step  = 25
     else:
          raise argparse.ArgumentError("model must be 'fnn', 'scnn', or 'mccn")    
     train.train(trainLoader, testLoader,
               model,
               max_lr=learning_rate,
               lr_mode='step',
               step=step,
               change_mo=False,
          #       epochs=40,
               epochs=epochs,
               print_freq=100,
               tune_alpha=True,
               regularizer = 0,
               log_wandb=log_wandb)


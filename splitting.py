'''
Code is adapted from the monotone operator network repository on GitHub
https://github.com/locuslab/monotone_op_net
'''

from typing import Tuple
import torch
import torch.nn as nn
from torch.autograd import Function
import utils
import time




class NEmonForwardStep(nn.Module):
    def __init__(self, linear_module, nonlin_module, alpha, tol=1e-4, max_iter=500, pretrain_iter=3, verbose=False):
        super().__init__()
        self.linear_module = linear_module
        self.nonlin_module = nonlin_module
        #self.alpha = 1/torch.max(torch.diag(torch.abs(self.linear_module.A.weight - torch.diag(torch.abs(self.linear_module.A.weight)@torch.ones(self.linear_module.A.weight.size[0])) + self.linear_module.m*torch.diag(torch.ones(self.linear_module.A.weight.size[0]))))) - 1e-8
        #self.alpha = 1/torch.max(torch.diag(torch.abs(self.linear_module.A.weight - torch.diag(torch.abs(self.linear_module.A.weight)@torch.ones(self.linear_module.A.weight.size()[0])) + self.linear_module.m*torch.diag(torch.ones(self.linear_module.A.weight.size()[0]))))) - 1e-8
        self.max_alpha = alpha
        self.backward_alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        print("MAX ITER:", max_iter)
        self.verbose = verbose
        self.stats = utils.SplittingMethodStats()
        self.save_abs_err = False
        self.pretrain = False
        self.pretrain_iter  = pretrain_iter

        self.last_eq = None # forward eq point, torch tensor
        self.last_back_eq = None
    def forward(self, x, max_iter = None, max_alpha = None):
        """ Forward pass of NEMON, find an equilibirum with averaged iterations"""
        

        
        start = time.time()

        # Reset alpha each time?
        running_alpha = self.max_alpha

        # Run the forward pass _without_ tracking gradients
        if not self.pretrain:
            with torch.no_grad():
                
                
                if self.last_eq is None or self.last_eq.shape[0] != x.shape[0]:# or True:
                    z = tuple(torch.zeros(s, dtype=x.dtype, device=x.device)
                        for s in self.linear_module.z_shape(x.shape[0]))
                else: 
                    z = tuple([self.last_eq])
                n = len(z)


                bias = self.linear_module.bias(x)

                # SEAN error const needs to be changed?
                err = 1
                it = 0
                errs = []

                
                while (err > self.tol and it < self.max_iter):
                    # Sasha
                    #print(len(z), z[0].shape)
                    zn = self.linear_module.multiply(*z)
                    zn = tuple((zn[i] + bias[i]) for i in range(n)) # TODO this needs to be vectorized
                    zn = self.nonlin_module(*zn)
                    zn = tuple((1 - running_alpha) * z[i] + running_alpha * zn[i] for i in range(n))
                    
                    #Original Mon
                    # zn = self.linear_module.multiply(*z)
                    # zn = tuple((1 - self.alpha) * z[i] + self.alpha * (zn[i] + bias[i]) for i in range(n))
                    # zn = self.nonlin_module(*zn)
                    if self.save_abs_err:
                        fn = self.nonlin_module(*self.linear_module(x, *zn))
                        err_new = sum((zn[i] - fn[i]).norm().item() / (zn[i].norm().item()) for i in range(n))
                        errs.append(err_new)
                    else:
                        err_new = sum((zn[i] - z[i]).norm().item() / (1e-6 + zn[i].norm().item()) for i in range(n))
                    
                    # DELETE:
                    if err_new > 0.85*err and running_alpha > 1e-3:
                        running_alpha /= 1.5
                    
                    err = err_new
                    z = zn
                    it = it + 1

            if self.verbose:
                print("Forward: ", it, err)

            # Run the forward pass one more time, tracking gradients, then backward placeholder
            zn = self.linear_module(x, *z)
            zn = self.nonlin_module(*zn)


            assert err < self.tol, 'Forward iteration not converged'
            
            zn = self.Backward.apply(self, *zn)

            # Uncomment to re-use fixed point
            #self.last_eq = zn[0].detach().clone()

            self.stats.fwd_iters.update(it)
            self.stats.fwd_time.update(time.time() - start)
            self.errs = errs
        else:
            z = tuple(torch.zeros(s, dtype=x.dtype, device=x.device)
                    for s in self.linear_module.z_shape(x.shape[0]))
            n = len(z)
            bias = self.linear_module.bias(x)

            # SEAN error const needs to be changed?
            err = 1
            it = 0
            errs = []

            
            while (err > self.tol and it < self.pretrain_iter):
                # Sasha
                zn = self.linear_module.multiply(*z)
                zn = tuple((zn[i] + bias[i]) for i in range(n)) # TODO this needs to be vectorized
                zn = self.nonlin_module(*zn)
                zn = tuple((1 - running_alpha) * z[i] + running_alpha * zn[i] for i in range(n))
                
                #Original Mon
                # zn = self.linear_module.multiply(*z)
                # zn = tuple((1 - self.alpha) * z[i] + self.alpha * (zn[i] + bias[i]) for i in range(n))
                # zn = self.nonlin_module(*zn)
                if self.save_abs_err:
                    fn = self.nonlin_module(*self.linear_module(x, *zn))
                    err_new = sum((zn[i] - fn[i]).norm().item() / (zn[i].norm().item()) for i in range(n))
                    errs.append(err_new)
                else:
                    err_new = sum((zn[i] - z[i]).norm().item() / (1e-6 + zn[i].norm().item()) for i in range(n))
                
                # DELETE:
                if err_new > 0.85*err and running_alpha > 1e-3:
                    running_alpha /= 1.5
                
                err = err_new
                z = zn
                it = it + 1

        if self.verbose:
            print("Forward: ", it, err)



        self.stats.fwd_iters.update(it)
        self.stats.fwd_time.update(time.time() - start)
        self.errs = errs

        return zn

    class Backward(Function):
        @staticmethod
        def forward(ctx, splitter, *z):
            ctx.splitter = splitter
            ctx.save_for_backward(*z)
            return z

        @staticmethod
        def backward(ctx, *g):
            start = time.time()
            sp = ctx.splitter
            
            n = len(g)
            z = ctx.saved_tensors
            j = sp.nonlin_module.derivative(*z)
            I = [j[i] == 0 for i in range(n)]
            d = [(1 - j[i]) / j[i] for i in range(n)]
            v = tuple(j[i] * g[i] for i in range(n))

            if sp.last_back_eq is None or sp.last_eq.shape[0] != g[0].shape[0]:
                u = tuple(torch.zeros(s, dtype=g[0].dtype, device=g[0].device)
                      for s in sp.linear_module.z_shape(g[0].shape[0]))
            else: 
                u = tuple([sp.last_back_eq])
            #sp.alpha = 0.11
            limiting_alpha = 0.2

            #running_alpha = sp.max_alpha
            # SEAN error const needs to be changed?
            err = 1.0
            it = 0
            errs = []
            while (err > sp.tol and it < sp.max_iter):
                un = sp.linear_module.multiply_transpose(*u)
                un = tuple((1 - limiting_alpha) * u[i] + limiting_alpha * un[i] for i in range(n))
                un = tuple((un[i] + limiting_alpha* (1 + d[i]) * v[i]) / (1 + limiting_alpha * d[i]) for i in range(n))
                for i in range(n):
                    un[i][I[i]] = v[i][I[i]]

                err_new = sum((un[i] - u[i]).norm().item() / (1e-6 + un[i].norm().item()) for i in range(n))
                errs.append(err_new)
                if err_new > 0.85*err and limiting_alpha > 1e-4:
                  limiting_alpha /= 1.5
                  
                err = err_new
                u = un
                it = it + 1

            # uncomment to re-use fized point as init
            #sp.last_back_eq = u[0].detach().clone()
            if sp.verbose:
                print("Backward: ", it, err, limiting_alpha)

            dg = sp.linear_module.multiply_transpose(*u)
            dg = tuple(g[i] + dg[i] for i in range(n))

            # assert err < sp.tol, f'Backward iteration not converged. err: {err}, tol: {sp.tol}'


            sp.stats.bkwd_iters.update(it)
            sp.stats.bkwd_time.update(time.time() - start)
            sp.errs = errs
            return (None,) + dg


class NEmonPeacemanRachford(nn.Module):

    def __init__(self, linear_module, nonlin_module, alpha=1.0, tol=1e-5, max_iter=50, verbose=False):
        super().__init__()
        self.linear_module = linear_module
        self.nonlin_module = nonlin_module
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.stats = utils.SplittingMethodStats()
        self.save_abs_err = False

    def forward(self, x):
        """ Forward pass of the MON, find an equilibirum with forward-backward splitting"""

        start = time.time()
        # Run the forward pass _without_ tracking gradients
        self.linear_module.init_inverse(1 + self.alpha, -self.alpha)
        with torch.no_grad():
            z = tuple(torch.zeros(s, dtype=x.dtype, device=x.device)
                      for s in self.linear_module.z_shape(x.shape[0]))
            u = tuple(torch.zeros(s, dtype=x.dtype, device=x.device)
                      for s in self.linear_module.z_shape(x.shape[0]))

            n = len(z)
            bias = self.linear_module.bias(x)

            err = 1.0
            it = 0
            errs = []
            while (err > self.tol and it < self.max_iter):
                u_12 = tuple(2 * z[i] - u[i] for i in range(n))
                z_12 = self.linear_module.inverse(*tuple(u_12[i] + self.alpha * bias[i] for i in range(n)))
                u = tuple(2 * z_12[i] - u_12[i] for i in range(n))
                zn = self.nonlin_module(*u)

                if self.save_abs_err:
                    fn = self.nonlin_module(*self.linear_module(x, *zn))
                    # TODO vectorize
                    err = sum((zn[i] - fn[i]).norm().item() / (zn[i].norm().item()) for i in range(n))
                    errs.append(err)
                else:
                    err = sum((zn[i] - z[i]).norm().item() / (1e-6 + zn[i].norm().item()) for i in range(n))
                z = zn
                it = it + 1

        if self.verbose:
            print("Forward: ", it, err)

        # Run the forward pass one more time, tracking gradients, then backward placeholder
        zn = self.linear_module(x, *z)
        zn2 = self.nonlin_module(*zn)
        with torch.no_grad():
            assert (torch.allclose(*z, *zn2))
        zn3 = self.Backward.apply(self, *zn2)

        with torch.no_grad():
            assert (torch.allclose(*zn2, *zn3))
        self.stats.fwd_iters.update(it)
        self.stats.fwd_time.update(time.time() - start)
        self.errs = errs
        return zn3

    class Backward(Function):
        @staticmethod
        def forward(ctx, splitter, *z):
            ctx.splitter = splitter
            ctx.save_for_backward(*z)
            return z

        @staticmethod
        def backward(ctx, *g):
            start = time.time()
            sp = ctx.splitter
            n = len(g)
            z = ctx.saved_tensors
            j = sp.nonlin_module.derivative(*z)
            I = [j[i] == 0 for i in range(n)]
            d = [(1 - j[i]) / j[i] for i in range(n)]
            v = tuple(j[i] * g[i] for i in range(n))

            z = tuple(torch.zeros(s, dtype=g[0].dtype, device=g[0].device)
                      for s in sp.linear_module.z_shape(g[0].shape[0]))
            u = tuple(torch.zeros(s, dtype=g[0].dtype, device=g[0].device)
                      for s in sp.linear_module.z_shape(g[0].shape[0]))

            err = 1.0
            errs=[]
            it = 0
            while (err >sp.tol and it < sp.max_iter):
                u_12 = tuple(2 * z[i] - u[i] for i in range(n))
                z_12 = sp.linear_module.inverse_transpose(*u_12)
                u = tuple(2 * z_12[i] - u_12[i] for i in range(n))
                zn = tuple((u[i] + sp.alpha * (1 + d[i]) * v[i]) / (1 + sp.alpha * d[i]) for i in range(n))
                for i in range(n):
                    zn[i][I[i]] = v[i][I[i]]

                err = sum((zn[i] - z[i]).norm().item() / (1e-6 + zn[i].norm().item()) for i in range(n))
                errs.append(err)
                z = zn
                it = it + 1

            if sp.verbose:
                print("Backward: ", it, err)

            dg = sp.linear_module.multiply_transpose(*zn)
            dg = tuple(g[i] + dg[i] for i in range(n))

            sp.stats.bkwd_iters.update(it)
            sp.stats.bkwd_time.update(time.time() - start)
            sp.errs = errs
            return (None,) + dg
# from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
from torch.distributions import normal, cauchy, laplace, uniform
import math 

class AttackBase(object):
    def __init__(self, model=None, norm=False, discrete=True, device=None):
        self.model = model
        self.norm = norm
        # Normalization are needed for CIFAR10, ImageNet
        if self.norm:
            self.mean = (0.4914, 0.4822, 0.2265)
            self.std = (0.2023, 0.1994, 0.2010)
        self.discrete = discrete
        self.device = device or torch.device("cuda:0")
        self.loss(device=self.device)

    def loss(self, custom_loss=None, device=None):
        device = device or self.device
        self.criterion = custom_loss or nn.CrossEntropyLoss()
        self.criterion.to(device)

    def perturb(self, x):
        raise NotImplementedError


    def normalize(self, x):
        if self.norm:
            y = x.clone().to(x.device) 
            if y.size(1) == 3:  # For RGB images (3 channels)
                y[:, 0, :, :] = (y[:, 0, :, :] - self.mean[0]) / self.std[0]
                y[:, 1, :, :] = (y[:, 1, :, :] - self.mean[1]) / self.std[1]
                y[:, 2, :, :] = (y[:, 2, :, :] - self.mean[2]) / self.std[2]
            elif y.size(1) == 1:  # For grayscale images (1 channel)
                y[:, 0, :, :] = (y[:, 0, :, :] - self.mean[0]) / self.std[0]
            else:
                raise ValueError("Unsupported number of channels: {}".format(y.size(1)))
            return y
        return x

    def inverse_normalize(self, x):    
        if self.norm:  
            y = x.clone().to(x.device)
            if y.size(1) == 3:  
                y[:, 0, :, :] = y[:, 0, :, :] * self.std[0] + self.mean[0]
                y[:, 1, :, :] = y[:, 1, :, :] * self.std[1] + self.mean[1]
                y[:, 2, :, :] = y[:, 2, :, :] * self.std[2] + self.mean[2]
            elif y.size(1) == 1:  
                y[:, 0, :, :] = y[:, 0, :, :] * self.std[0] + self.mean[0]
            else:
                raise ValueError("Unsupported number of channels: {}".format(y.size(1)))
        
            return y
        return x


    def discretize(self, x):
        return torch.round(x * 255) / 255

    # Change this name as "projection"
    def clamper(self, x_adv, x_nat, bound=None, metric="inf", inverse_normalized=False):
        if not inverse_normalized:
            x_adv = self.inverse_normalize(x_adv)
            x_nat = self.inverse_normalize(x_nat)
        if metric == "inf":
            clamp_delta = torch.clamp(x_adv - x_nat, -bound, bound)
        else:
            clamp_delta = x_adv - x_nat
            for batch_index in range(clamp_delta.size(0)):
                image_delta = clamp_delta[batch_index]
                image_norm = image_delta.norm(p=metric, keepdim=False)
                # TODO: channel isolation?
                if image_norm > bound:
                    clamp_delta[batch_index] /= image_norm
                    clamp_delta[batch_index] *= bound
        x_adv = x_nat + clamp_delta
        x_adv = torch.clamp(x_adv, 0., 1.)
        return self.normalize(self.discretize(x_adv)).clone().detach().requires_grad_(True)


class FGSM(AttackBase):
    def __init__(self, model=None, bound=None, norm=False, random_start=False, discrete=True, device=None, **kwargs):
        super(FGSM, self).__init__(model, norm, discrete, device)
        self.bound = bound
        self.rand = random_start
        
    # @overrides
    def perturb(self, x, y, model=None, bound=None, device=None, dist='normal',
     lamda =0, l1_norm= False, current_iter=None,scaling=None, decay_rate=0.01, 
     total_iterations=100,  gamma = 0, standard_sgd =False, **kwargs):
        criterion = self.criterion
        model = model or self.model
        bound = bound or self.bound
        device = device or self.device

        model.zero_grad()
        x_nat = self.inverse_normalize(x.detach().clone().to(device))
        x_adv = x.detach().clone().requires_grad_(True).to(device)
        
        pred = model(x_adv)
        if criterion.__class__.__name__ == "NLLLoss":
            pred = F.softmax(pred, dim=-1)
        loss = criterion(pred, y)
        loss.backward()
        
        # select a distribution
        if dist == 'normal':
            z = torch.randn(x_adv.shape, device=device)
        elif dist == 'cauchy':
            z = cauchy.Cauchy(0, 1).sample(x_adv.shape).to(device)
        elif dist == 'laplacian':
            z = laplace.Laplace(0, 1).sample(x_adv.shape).to(device)
        elif dist == 'uniform':
            z = uniform.Uniform(-1, 1).sample(x_adv.shape).to(device)
        else:
            raise ValueError(f"Unsupported distribution: {dist}")

        if lamda > 0:
            if scaling == 'inverse':
                lamda = inverse_scaling(lamda, current_iter)
            elif scaling == 'exponential_decay':
                lamda = exponential_decay_scaling(lamda, current_iter, decay_rate)
            elif scaling == 'linear_decay':
                lamda = linear_decay_scaling(lamda, current_iter, total_iterations)

            
            grad_sign = (x_adv.grad.data.detach()+ lamda*(x_adv.data.detach()-x_nat.data.detach()) + gamma * z).sign()

        # if not using closeness regularizaition, scale z by gamma. gamma is 0 in SOTA method by chen et al 
        else:    
            grad_sign = (x_adv.grad.data.detach() + gamma * z).sign()

        x_adv = self.inverse_normalize(x_adv) + grad_sign * bound
        x_adv = self.clamper(x_adv, x_nat, bound=bound, inverse_normalized=True)

        return x_adv.detach()


def inf_generator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def inverse_scaling(lamda, iteration):
    return lamda / iteration

def exponential_decay_scaling(lamda, iteration, decay_rate=0.01):
    return lamda * math.exp(-decay_rate * iteration)

def linear_decay_scaling(lamda, iteration, total_iterations):
    return lamda * (1 - (iteration / total_iterations))





class LinfPGD(AttackBase):
    def __init__(self, model=None, bound=None, step=None, iters=None, norm=False, random_start=False, discrete=True,
                 device=None,  gamma = 0, sgld =True, **kwargs):
        super(LinfPGD, self).__init__(model, norm, discrete, device)
        self.bound = bound
        self.step = step
        self.iter = iters
        self.rand = random_start
        self.sgld = sgld
        self.gamma = gamma
    

    # @overrides
    def perturb(self, x, y, target_y=None, model=None, bound=None, step=None, iters=None, x_nat=None, device=None, dist = 'normal',
                **kwargs):
        criterion = self.criterion
        model = model or self.model
        bound = bound or self.bound
        step = step or self.step
        iters = iters or self.iter
        device = device or self.device

        model.zero_grad()
        if x_nat is None:
            x_nat = self.inverse_normalize(x.detach().clone().to(device))
        else:
            x_nat = self.inverse_normalize(x_nat.detach().clone().to(device))
        x_adv = x.detach().clone().requires_grad_(True).to(device)
        if self.rand:
            rand_perturb_dist = distributions.uniform.Uniform(-bound, bound)
            rand_perturb = rand_perturb_dist.sample(sample_shape=x_adv.shape).to(device)
            x_adv = self.clamper(self.inverse_normalize(x_adv) + rand_perturb, self.inverse_normalize(x_nat),
                                 bound=bound, inverse_normalized=True)
            if self.discretize:
                x_adv = self.normalize(self.discretize(x_adv)).detach().clone().requires_grad_(True)
            else:
                x_adv = self.normalize(x_adv).detach().clone().requires_grad_(True)

        for i in range(iters):
            adv_pred = model(x_adv)
            ori_pred = model(x)
            delta_pred = adv_pred - ori_pred
            if criterion.__class__.__name__ == "NLLLoss":
                delta_pred = F.log_softmax(delta_pred, dim=-1)
            # loss =   0.1*criterion(pred, target_y) - criterion(pred, original_y)
            if target_y is not None:
                # loss = criterion(adv_pred, y)
                loss = - criterion(delta_pred, target_y)  # + 0.01*criterion(delta_pred, y)
            else:
                loss = criterion(adv_pred, y)
            loss.backward()


            if self.gamma != 0:
                if dist == 'normal':
                    z = torch.randn(x_adv.shape, device=device)
                elif dist == 'cauchy':
                    z = cauchy.Cauchy(0, 1).sample(x_adv.shape).to(device)
                elif dist == 'laplacian':
                    z = laplace.Laplace(0, 1).sample(x_adv.shape).to(device)
                elif dist == 'uniform':
                    z = uniform.Uniform(-1, 1).sample(x_adv.shape).to(device)
                else:
                    raise ValueError(f"Unsupported distribution: {dist}")
                
                grad_sign = (x_adv.grad.data.detach() + self.gamma * z).sign()
                x_adv = self.inverse_normalize(x_adv) + grad_sign * step
                x_adv = self.clamper(x_adv, x_nat, bound=bound, inverse_normalized=True)
                model.zero_grad()

            else:      
                grad_sign = x_adv.grad.data.detach().sign()
                x_adv = self.inverse_normalize(x_adv) + grad_sign * step
                x_adv = self.clamper(x_adv, x_nat, bound=bound, inverse_normalized=True)
                model.zero_grad()
                
            x_adv = self.clamper(x_adv, x_nat, bound=bound, inverse_normalized=True)

        return x_adv.detach().to(device)

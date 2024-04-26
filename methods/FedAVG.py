import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.distributions as dist

import copy
import numpy as np
from collections import defaultdict, OrderedDict

from methods.base import *
from util import *


class Model(Base):
    def __init__(self, args):
        self.probabilistic = False
        super(Model, self).__init__(args)

    def train_client(self,loader,steps=1):
        self.train()
        lossMeter = AverageMeter()
        accMeter = AverageMeter()
        for step in range(steps):
            x, y = next(iter(loader))
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = F.cross_entropy(logits,y)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            acc = (logits.argmax(1)==y).float().mean()
            lossMeter.update(loss.data,x.shape[0])
            accMeter.update(acc.data,x.shape[0])

        return {'acc': accMeter.average(), 'loss': lossMeter.average()}


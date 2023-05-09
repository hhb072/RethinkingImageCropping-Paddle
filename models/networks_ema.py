import numbers
import copy

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from functools import wraps


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# class MemoryAE(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#
#         self.netG = NetG(config)
#
#         self.netD = Discriminator(config.ndf)
#
#         self.netT = None
#         self.ema_updater = EMA(config.moving_average_decay)
#         self.produce_pseudo_label(torch.zeros(1,3,32,32))
#
#
#     def forward(self, x):
#         pass
#
#
#     @singleton('netT')
#     def _get_target_network(self):
#         netT = copy.deepcopy(self.netG)
#         return netT
#
#     def reset_moving_average(self):
#         del self.netT
#         self.netT = None
#
#     def update_moving_average(self):
#         assert self.netT is not None, 'target network has not been created yet'
#         update_moving_average(self.ema_updater, self.netT, self.netG)
#
#     def produce_pseudo_label(self, x):
#         with torch.no_grad():
#             netT = self._get_target_network()
#             return self.netT(x)
#
#
#
# if config.pretrained:
#     state = torch.load(config.pretrained)
#     model.load_state_dict(state['model'], strict=False)
#     model.reset_moving_average()
#     model.netT = model._get_target_network()
#
#
#
# optimizerG.zero_grad()
# loss.backward()
# optimizerG.step()
#
# model.update_moving_average()
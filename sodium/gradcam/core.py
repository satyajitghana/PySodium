from collections import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

'''
majority of the code taken from: https://github.com/kazuto1011/grad-cam-pytorch/blob/master/grad_cam.py

Note:
Images are stored as shape [BATCH B, CHANNEL C, HEIGHT H, WIDTH W]

> from pytorch forum
> No, we only support NCHW format. You can use .permute 20.1k to swap the axis.
'''


class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()

        # assuming all parameters are on the same device
        self.device = next(model.parameters()).device

        # save the model
        self.model = model

        # a set of hook function handlers
        self.handlers = []

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        # get H X W
        self.image_shape = image.shape[2:]

        # apply the model
        self.logits = self.model(image)

        # get the loss by converging along all the channels, dim = CHANNEL
        # sum along CHANNEL is going to be 1, softmax does that
        self.probs = F.softmax(self.logits, dim=1)

        # ordered results
        return self.probs.sort(dim=1, descending=True)

    def backward(self, ids):
        '''Class-specific backpropagation'''

        # convert the class id to one hot vector
        one_hot = self._encode_one_hot(ids)

        # zero out the gradients
        self.model.zero_grad()

        # calculate the gradient wrt to the class activations
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        '''Remove all the forward/backward hook functions'''
        for handle in self.handlers:
            handle.remove()


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(
                    module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(
                    module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError(f'Invalid layer name: {target_layer}')

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        # rescale features between 0 and 1
        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam

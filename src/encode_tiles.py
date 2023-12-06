import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from skimage.color import rgb2gray
from skimage.filters import  threshold_otsu
from skimage.morphology import square
if os.environ['USE_TRANSPATH'] == 'True':
    from timm.models.layers.helpers import to_2tuple
    import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, resnet50
from torchvision.transforms import ToTensor

from .downloads import (download_ctranspath, download_moco,
                        download_pca_ctranspath)
from .tile_slide import SlideTileDataset


def load_moco_model(moco_weights_path, model_name='resnet18'):
    """
    Loads a resnet with moco pretrained weights.
    Args:
        moco_weights_path (str): Path to moco weights.
        model_name (str): Name of the model.
    Returns:
        model (torch.nn.Module): Model with moco weights.
    """

    model = eval(model_name)()
    checkpoint = torch.load(moco_weights_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    for name, param in model.named_parameters():
        if 'fc' in name:
            continue
        assert (param == state_dict[name]).all().item(), 'Weights not loaded properly'
    print('Loaded the weigths properly.')
    return model

def ctranspath():
    """
    Code taken from https://github.com/Xiyue-Wang/TransPath
    """
    model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
    return model

class ConvStem(nn.Module):
    """
    Code taken from https://github.com/Xiyue-Wang/TransPath
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

def load_transpath_model(transpath_weights_path):
    """
    Code taken from https://github.com/Xiyue-Wang/TransPath
    """
    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load(transpath_weights_path)
    model.load_state_dict(td['model'], strict=True)
    return model

class ModelWrapper:
    """Wraps a model so that forward it outputs the right embeddings,
    depending on the type of model"""
    def __init__(self, model, last_layer=True, pca=None):
        self.model = model
        self.last_layer = last_layer
        self.pca = pca
        if not last_layer:
            self.hook = [0]
            def hook_l3(m, i, o):
                self.hook[0] = o
            self.model.layer3.register_forward_hook(hook_l3)

    def __call__(self, x):
        embeddings = self.model(x)
        if not self.last_layer:
            embeddings = torch.mean(self.hook[0], dim=(2, 3))
        embeddings = embeddings.squeeze().detach().cpu().numpy()
        if self.pca is not None:
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1,-1)
            embeddings = self.pca.transform(embeddings)[:,:256]
        return embeddings

def get_tile_encoder(model, device):
    model_path = eval(f'download_{model}()')
    if model == 'moco':
        model = load_moco_model(model_path)
        model.eval().to(device)
        return ModelWrapper(model, last_layer=False)
    elif model == 'ctranspath':
        model = load_transpath_model(model_path)
        model.eval().to(device)
        pca = download_pca_ctranspath()
        pca = np.load(pca, allow_pickle=True).item()
        return ModelWrapper(model, last_layer=True, pca=pca)

def get_embeddings(model, image_path, N_ensemble=500, magnification_tile=10, device='cpu', n_tiles_during_training=5, store_intermediate=None):
    """
    Computes the embeddings of a slide.
    Args:
        model (str): Name of the model. either moco or ctranspath
        image_path (str): Path to the slide.
        N_ensemble (int): Number of tiles to use for the ensemble.
        magnification_tile (int): Magnification of the tiles.
        device (str): Device to use.
        n_tiles_during_training (int): Number of tiles used during training of the gigassl model.
        store_intermediate (str): Path to store the intermediate images during tiling.
        
    Returns:
        embeddings (np.array): Embeddings of the slide.
        xys (np.array): Coordinates of the tiles.
    """
    data = SlideTileDataset(image_path, N_ensemble=N_ensemble, magnification_tile=magnification_tile, n_tiles_during_training=n_tiles_during_training, store_intermediate=store_intermediate)
    if len(data) == 0:
        return None, None
    num_cpus = len(os.sched_getaffinity(0))
    dataloader = DataLoader(data, batch_size=min(100, len(data)), shuffle=False, num_workers=num_cpus)
    embeddings = []
    xys = []
    for batch in dataloader:
        with torch.no_grad():
            im, xy = batch
            im = im.to(device)
            xys.append(xy)
            e = model(im)
            e = e.reshape((-1, 256))
            embeddings.append(e)
    embeddings = np.concatenate(embeddings)
    xys = torch.concatenate(xys)
    del dataloader, data
    del batch
    return embeddings, xys


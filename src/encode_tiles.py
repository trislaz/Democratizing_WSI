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
if (os.environ['USE_GIGAPATH'] == 'True') or (os.environ['USE_UNI'] == 'True'):
    import timm

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, resnet50
from torchvision.transforms import ToTensor
from transformers import ViTModel

from .downloads import download_item
# (download_ctranspath, download_moco,
#                         download_pca_ctranspath, download_pca_phikon, download_phikon, 
#                         download_gigapath, download_pca_gigapath, download_pca_uni, download_uni)
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

def load_phikon_model(path_placeholder):
    """
    Loads the phikon model.
    """
    model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
    return model

def load_optimus_model(path=None):
    """
    Loads the optimus model.
    """
    if path is None:
        PATH_TO_CHECKPOINT = "/cluster/CBIO/home/tlazard/.cache/dl_models/bioptimus/h-optimus-0.pth"  # Path to the downloaded checkpoint.

        params = {
            'patch_size': 14,
            'embed_dim': 1536,
            'depth': 40,
            'num_heads': 24,
            'init_values': 1e-05,
            'mlp_ratio': 5.33334,
            'mlp_layer': functools.partial(
                timm.layers.mlp.GluMlp, act_layer=torch.nn.modules.activation.SiLU, gate_last=False
            ),
            'act_layer': torch.nn.modules.activation.SiLU,
            'reg_tokens': 4,
            'no_embed_class': True,
            'img_size': 224,
            'num_classes': 0,
            'in_chans': 3
        }

        model = timm.models.VisionTransformer(**params)
        model.load_state_dict(torch.load(PATH_TO_CHECKPOINT, map_location="cpu"))
        model.eval()
    return model

def load_gigapath_model(path_placeholder=None):
    """
    @https://github.com/prov-gigapath/prov-gigapath
    """
    if path_placeholder is None:
        tile_encoder = timm.create_model('hf_hub:prov-gigapath/prov-gigapath', pretrained=True)
        tile_encoder.eval()
    return tile_encoder

def load_uni_model(path='/cluster/CBIO/home/tlazard/.cache/dl_models/uni/pytorch_model.bin'):
    """
    @https://huggingface.co/MahmoodLab/UNI
    """
    if path is None:
        model = timm.create_model(
    "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
)
        model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
        model.eval()
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
    def __init__(self, model, last_layer=True, pca=None, phikon=False):
        self.model = model
        self.last_layer = last_layer
        self.phikon = phikon
        if phikon:
            self.last_layer=True
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
        if self.phikon:
            embeddings = embeddings.last_hidden_state[:,0,:]
        embeddings = embeddings.squeeze().detach().cpu().numpy()
        if self.pca is not None:
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1,-1)
            embeddings = self.pca.transform(embeddings)[:,:256]
        return embeddings
def get_tile_encoder(model, device):
    model_path = download_item(model)
    if model == 'moco':
        model = load_moco_model(model_path)
        model.eval().to(device)
        return ModelWrapper(model, last_layer=False)
    elif model == 'ctranspath':
        model = load_transpath_model(model_path)
        model.eval().to(device)
        pca_path = download_item('pca_ctranspath')
        pca = np.load(pca_path, allow_pickle=True).item()
        return ModelWrapper(model, last_layer=True, pca=pca)
    elif model == 'phikon':
        model = load_phikon_model(model_path)
        model.eval().to(device)
        pca_path = download_item('pca_phikon')
        pca = torch.load(pca_path)
        return ModelWrapper(model, last_layer=True, pca=pca, phikon=True)
    elif model == "gigapath":
        model = load_gigapath_model(model_path)
        model.eval().to(device)
        pca_path = download_item('pca_gigapath')
        pca = torch.load(pca_path)
        return ModelWrapper(model, last_layer=True, pca=pca, phikon=False)
    elif model == "uni":
        model = load_gigapath_model()
        model.eval().to(device)
        pca_path = download_item('pca_uni')
        pca = torch.load(pca_path)
        return ModelWrapper(model, last_layer=True, pca=pca, phikon=False)
    elif model == "optimus":
        model = load_optimus_model()
        model.eval().to(device)
        pca_path = download_item('pca_optimus')
        pca = torch.load(pca_path)
        return ModelWrapper(model, last_layer=True, pca=pca, phikon=False)
    
def get_embeddings(model, image_path, N_ensemble=500, magnification_tile=10, device='cpu', n_tiles_during_training=5, store_intermediate=None):
    """
    Computes the embeddings of a slide.
    Args:
        model (str): Name of the tile encoding model. either moco or ctranspath or phikon or gigapath or uni or optimus
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
    data = SlideTileDataset(image_path, N_ensemble=N_ensemble, magnification_tile=magnification_tile, n_tiles_during_training=n_tiles_during_training, store_intermediate=store_intermediate, model=model)
    if len(data) == 0:
        return None, None
    num_cpus = len(os.sched_getaffinity(0))
    dataloader = DataLoader(data, batch_size=min(100, len(data)), shuffle=False, num_workers=num_cpus)
    embeddings = []
    xys = []
    for batch in dataloader:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.inference_mode():
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


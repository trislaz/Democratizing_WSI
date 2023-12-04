import os
from argparse import Namespace
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from skimage.color import rgb2gray
from sklearn.preprocessing import Normalizer
from torchvision import transforms
from torchvision.models import resnet18, resnet50
from torchvision.transforms import ToTensor
from tqdm import tqdm

from .downloads import (download_gigassl_mlp_ctranspath,
                        download_gigassl_mlp_moco,
                        download_gigassl_scm_ctranspath,
                        download_gigassl_scm_moco)
from .encode_tiles import (ModelWrapper, get_embeddings, get_tile_encoder,
                           load_moco_model)
from .networks import FullSparseConvMIL, ResidualMLP
from .tile_slide import SlideTileDataset

from functools import wraps

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

def encode_wsi_from_embeddings(model, hook, embedding, xy, device, n_tiles_during_training=5):
    im = torch.Tensor(embedding)
    repre = []
    for rep in range(0, im.shape[0] , n_tiles_during_training):
        with torch.no_grad():
            start = rep 
            end = rep + min(n_tiles_during_training, im.shape[0] - rep)
            if end >= im.shape[0]:
                end = im.shape[0]
            im_s = im[start:end, :].unsqueeze(0).to(device)
            xy_s = torch.Tensor(xy)[start:end, :].unsqueeze(0).to(device)
            _ = model((im_s, xy_s))
            repre.append(hook.item)
    repre = np.vstack([x[0].cpu().numpy() for x in repre])
    repre = np.mean(repre, axis=0)
    norm = Normalizer()
    repre = norm.fit_transform(repre.reshape(1, -1))
    return np.squeeze(repre)

def load_pretrained_model(model_path, gigassl_type):
    builder = FullSparseConvMIL if gigassl_type == 'scm' else ResidualMLP
    ckpt = torch.load(model_path, map_location='cpu')
    config = Namespace(**ckpt['config'])
    model = builder(config)
    state_dict = ckpt['state_dict']
    state_dict = {k.replace('backbone.mil.', ''):w for k,w in state_dict.items() if k.startswith('backbone') and not 'classifier' in k}
    model.load_state_dict(state_dict, strict=False)
    for name, param in model.named_parameters():
        if 'classifier' in name:
            continue
        assert (param == state_dict[name]).all().item(), 'Weights not loaded properly'
    print('Loaded the weigths properly.')
    return model

def get_model_and_hook(tile_encoder_type, gigassl_type):
    path = eval(f'download_gigassl_{gigassl_type}_{tile_encoder_type}()')
    model = load_pretrained_model(path, gigassl_type)
    hooker = type('hooker', (), {'item':None})()
    def hook(m, i, o):
        hooker.item = i[0].cpu()
    #Here, bit dirty...
    if gigassl_type == 'scm':
        handle = model.net.linear_classifier.register_forward_hook(hook)
    elif gigassl_type == 'mlp':
        handle = model.classifier.register_forward_hook(hook)
    return model, hooker

@timing
def encode_image(tile_encoder_type, gigassl_type, image_path, N_ensemble, magnification_tile=10, last_layer=False, n_tiles_during_training=5, store_intermediate=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tile_encoder = get_tile_encoder(tile_encoder_type, device)

    gigassl, hook_giga = get_model_and_hook(tile_encoder_type, gigassl_type)
    gigassl.eval().to(device)

    image_path = Path(image_path)
    if image_path.is_dir():
        images_paths = [x for x in image_path.iterdir() if x.suffix in ['.svs', '.ndpi', '.tiff', '.tif']]
    else:
        images_paths = [image_path]
    
    dico_embs = {}
    for o, image_path in tqdm(enumerate(images_paths)):
        embeddings_tiles, xy_tiles = get_embeddings(tile_encoder, str(image_path), N_ensemble=N_ensemble, magnification_tile=magnification_tile, device=device, store_intermediate=store_intermediate)
        if embeddings_tiles is None:
            print(f'No tiles for {image_path}')
            dico_embs[image_path.stem] = None
            continue
        embedding_gigassl = encode_wsi_from_embeddings(gigassl, hook_giga, embeddings_tiles, xy_tiles, device, n_tiles_during_training=n_tiles_during_training)
        dico_embs[image_path.stem] = embedding_gigassl
    return dico_embs

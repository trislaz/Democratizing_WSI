import itertools
import math
import os
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
import torch
from PIL import Image, ImageDraw
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing, opening, square
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, resnet50
from torchvision.transforms import ToTensor


class SlideTileDataset(Dataset):
    mag_level0 = 40
    ds_per_level = {'.svs': 4, '.ndpi':2, '.tiff':2, '.tif':2}
    final_tile_size = 224
    normalization_params = {
        'default': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        'optimus': {'mean': (0.707223, 0.578729, 0.703617), 'std': (0.211883, 0.230117, 0.177517)}
    }

    def __init__(self, image_path, N_ensemble=None, magnification_tile=10, n_tiles_during_training=5, store_intermediate=None, model='default'):
        self.n_tiles_during_training = n_tiles_during_training # Models are trained with 5 tiles per slide
        self.max_tiles_per_slide = self.n_tiles_during_training * N_ensemble if N_ensemble is not None else None
        self.magnification_tile = magnification_tile
        self.mask_level = -1
        self.image_path = image_path
        self.level_tile = self._get_level(self.magnification_tile, mag_level0=self.mag_level0)
        self.mask_tolerance = 0.9
        self.image = openslide.open_slide(self.image_path)
        self.thumbnail = self._get_thumbnail()
        self.model = model

        if store_intermediate is not None:
            self.store = Path(store_intermediate)
            image_stem = Path(image_path).stem
            self.store_tiles = self.store / f'{image_stem}_tiles'
            self.store_tiles.mkdir(exist_ok=True, parents=True)
            self.store_masked_thumbnail = self.store / f'{image_stem}_masked_thumbnail.png'
        else:
            self.store = None
            self.store_tiles = None
            self.store_masked_thumbnail = None

        self.params = self._prepare_params()
        self.transforms = self._get_transforms()
        

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        param = self.params[idx]
        row, col, h, w = param[:4]
        tile = self.image.read_region(location=(col, row), level=self.level_tile, size=(min(self.final_tile_size, self.image.dimensions[0] - col), min(self.final_tile_size, self.image.dimensions[1] - row))).convert('RGB')
        if self.store_tiles is not None:
            tile.save(self.store_tiles / f'{Path(self.image_path).stem}_{idx}.png')
        tile = self.transforms(tile)
        return tile, torch.Tensor(np.array([col, row]))

    def _get_transforms(self):
        norm_params = self.normalization_params.get(self.model, self.normalization_params['default'])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_params['mean'], std=norm_params['std'])
        ])

    def _get_thumbnail(self):
        thumbnail = self.image.get_thumbnail(self.image.level_dimensions[self.mask_level])
        return thumbnail
 
    def _make_masked_thumbnail(self, params):
        draw = ImageDraw.Draw(self.thumbnail)
        ds = self.image.level_downsamples[self.mask_level]
        for index, param in enumerate(params):
            row, col, h, w = param[:4]
            scaled_row, scaled_col = row // ds, col // ds
            scaled_h, scaled_w = h // ds, w // ds
            draw.rectangle((scaled_col, scaled_row, scaled_col + scaled_w, scaled_row + scaled_h), outline='red')
            draw.text((scaled_col, scaled_row), str(index), fill='red')

    def _prepare_params(self):
        dico = self._get_clean_grid(self.image, np.array(self.thumbnail), self.final_tile_size, level=self.level_tile, mask_tolerance=self.mask_tolerance)
        if isinstance(self.max_tiles_per_slide, int):
            indices = np.random.choice(len(dico['params']), min(self.max_tiles_per_slide, len(dico['params'])), replace=False)
            params = [dico['params'][i] for i in indices]
        else:
            params = dico['params']
        if self.store_tiles is not None:
            self._make_masked_thumbnail(params)
            self.thumbnail.save(self.store_masked_thumbnail)
        return params

    def _get_level(self, magnification_tile, mag_level0):
        ext = Path(self.image_path).suffix
        return int(math.log(mag_level0 / magnification_tile, self.ds_per_level[ext]))

    def _get_clean_grid(self, image, thumbnail,  tile_size, level, mask_tolerance):
        """
        x = columns 
        y = rows
        """
        image_height, image_width = self.image.dimensions[1], self.image.dimensions[0]
        mask_ds = int(self.image.level_downsamples[self.mask_level])
        mask = self._make_auto_mask(thumbnail)
        size_at_0 = tile_size * (2 ** level)
        grid = self._grid_blob(image, (0, 0), (image_height, image_width), (size_at_0, size_at_0), 0)
        grid = [(x[0], x[1], size_at_0, size_at_0, 0)  for x in grid if self._check_coordinates(x[0], x[1], (size_at_0, size_at_0), mask, mask_ds, mask_tolerance)]
        dico = {'params': grid, 'mask': mask}
        return dico 

    def _make_auto_mask(self, thumbnail):
        """make_auto_mask. Create a binary mask from a downsampled version
        of a WSI. Uses the Otsu algorithm and a morphological opening.
        """
        im = np.array(thumbnail)[:, :, :3]
        im_gray = rgb2gray(im)
        #im_gray = self._clear_border(im_gray, prop=10)
        size = im_gray.shape
        im_gray = im_gray.flatten()
        pixels_int = im_gray[np.logical_and(im_gray > 0.02, im_gray < 0.98)]
        t = threshold_otsu(pixels_int)
        mask = opening(closing((im_gray < t).reshape(size), square(2)), square(2))
        return mask

    @staticmethod
    def _clear_border(mask, prop):
        r, c = mask.shape
        pr, pc = r // prop, c // prop
        mask[:pr, :] = 0
        mask[r - pr :, :] = 0
        mask[:, :pc] = 0
        mask[:, c - pc :] = 0
        return mask

    def _grid_blob(self, slide, point_start, point_end, patch_size,
              analyse_level):
        """
        Forms a uniform grid starting from the top left point point_start
        and finishes at point point_end of size patch_size at level analyse_level
        for the given slide.
        Args:
            slide : String or open_slide object. 
            point_start : Tuple like object of integers of size 2.
            point_end : Tuple like object of integers of size 2. (x,y) = (row, col) = (height, width)
            patch_size : Tuple like object of integers of size 2.
            analse_level : Integer. Level resolution to use for extracting the tiles.
        Returns:
            List of coordinates of grid.
        """
        patch_size_0 = patch_size
        size_x, size_y = patch_size_0
        list_col = range(point_start[1], point_end[1], size_x)
        list_row = range(point_start[0], point_end[0], size_y)
        return list(itertools.product(list_row, list_col))

    def _check_coordinates(self, row, col, patch_size, mask, mask_downsample, mask_tolerance=0.8):
        """
        Checks if the patch at coordinates x, y in res 0 is valid.
        Args:
            x : Integer. x coordinate of the patch.
            y : Integer. y coordinate of the patch.
            patch_size : Tuple of integers. Size of the patch.
            mask : Numpy array. Mask of the slide.
            mask_downsample : Integer. Resolution of the mask.
        Returns:
            Boolean. True if the patch is valid, False otherwise.
        """
        col_0, row_0 = col, row
        col_1, row_1 = col + patch_size[0], row + patch_size[1]
        ## Convert coordinates to mask_downsample resolution
        col_0, row_0 = col_0 // mask_downsample, row_0 // mask_downsample
        col_1, row_1 = col_1 // mask_downsample, row_1 // mask_downsample
        if col_0 < 0 or row_0 < 0 or row_1 > mask.shape[0] or col_1 > mask.shape[1]:
            return False
        mask_patch = mask[row_0:row_1, col_0:col_1]
        if mask_patch.sum() <= mask_tolerance * np.ones(mask_patch.shape).sum():
            return False
        return True


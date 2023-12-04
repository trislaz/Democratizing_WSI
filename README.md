# Democratizing computational pathology: optimized Whole Slide Image representations for The Cancer Genome Atlas

Companion package for the paper "Democratizing computational pathology: optimized Whole Slide Image representations for The Cancer Genome Atlas".
This package will allow you to encode Whole Slide Images into their Giga-SSL representation (1 feature vector) with **one command line**.

# Usage

Nothing easier. Just use the `main.py` script; it will handle the necessary model downloads for you.

```bash
python main.py --input /path/to/WSI/folder/ --output tcga_brca/ --tile_encoder_type moco
```

Note: `--input` is the path of either a WSI or a folder containing WSI (in .ndpi, .svs or .tif format)

## Arguments of main.py

- `--input`: path of either a WSI or a folder containing WSI (in .ndpi, .svs or .tif format) (required)
- `--output`: path of the output folder (required)
- `--tile_encoder_type`: type of tile encoder to use (default: moco, available: moco, ctranspath)
- `--gigassl_type`: type of gigassl model (default: scm, available: scm, mlp)
- `--N_ensemble`: number of WSI views to ensemble (default: 100). Usually, the more the better, but the more, the heavier computationally.
- `--store_intermediate`: path of the folder where to store intermediate results (default: None). Intermediate results include tissue mask, tiles localizations as well as the used tiles themselves.

# Installation

## Basic requirements

We recommend using conda to install the necessary dependencies. If you don't have conda installed, please follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).
Then, in this repository, run:

```bash
conda env create -f requirements.yml && conda activate gigassl
```

With these requirements you will be able to run gigassl models of type `mlp` using the `moco` encoder.
They provide a quick way to test the package, as well as good WSIs embeddings (even if a bit less performant than the ones obtained with `scm` models).

## Requirements to run gigassl models of type `scm`

`scm` refers to SparseConvMIL-related models. These are the ones evaluated in the paper.

After having installed the basic requirements, you will need to install the [SparseConvNet](https://github.com/facebookresearch/SparseConvNet), following the instruction on its github repository.
In their setup instructions, do not forget to have at least one GPU available when launching `bash develop.sh`, otherwise the GPU version of the package will not be installed.

## Requirements to run gigassl trained on top of CTransPath embeddings

`scm` and `mlp` GigaSSL models have been trained on both MoCo (in-house model) and CTransPath embeddings.
Performances using the CTransPath embeddings are better than the ones using MoCo embeddings.
As stated in the [CTransPath original package](https://github.com/Xiyue-Wang/TransPath), you will have to:

* Download [a modified timm library here](https://drive.google.com/file/d/1JV7aj9rKqGedXY1TdDfi3dP07022hcgZ/view?usp=sharing)
* install it: `pip install timm-0.5.4.tar`

# Outputs

The `main.py` script will output a file `features_{tile_encoder_type}_{gigassl_type}.csv` in the output folder.
It is a dictionary mapping WSI names to their corresponding GigaSSL features.

```
import numpy as np
embeddings = np.load('features_moco_scm.npy', allow_pickle=True).item()
```

Finally, the file `test_classification.py` downloads TCGA embeddings and performs classification tasks on them.
It provides the classification implementation used in the article.

Here are the results of 10-fold CV AUCs obtained with the 'scm' model trained on top of either CTranspath or MoCo embeddings. These are classification perfomances of **simple logistic regression !**

|                 | GigaSSL + Moco | GigaSSL + Moco | GigaSSL + CTransPath | GigaSSL + CTransPath |
| --------------- | -------------- | -------------- | -------------------- | -------------------- |
|                 | Mean           | Std            | Mean                 | Std                  |
| brca_molectype  | 0.929          | 0.03           | 0.942                | 0.036                |
| lung            | 0.962          | 0.015          | 0.975                | 0.011                |
| brca_mhrd       | 0.788          | 0.044          | 0.805                | 0.034                |
| brca_histotypes | 0.929          | 0.027          | 0.944                | 0.031                |
| kidney          | 0.984          | 0.01           | 0.992                | 0.007                |


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
- `--tile_encoder_type`: type of tile encoder to use (default: moco, available: moco, [ctranspath](https://github.com/Xiyue-Wang/TransPath), [phikon](https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v1))
- `--gigassl_type`: type of gigassl model (default: scm, available: scm, mlp)
- `--N_ensemble`: number of WSI views to ensemble (default: 100). Usually, the more the better, but the more, the heavier computationally.
- `--store_intermediate`: path of the folder where to store intermediate results (default: None). Intermediate results include tissue mask, tiles localizations as well as the used tiles themselves.

# Parallel computing

Fill the `slurm_encode.sh` to launch the encodings on multiple nodes.

# Test

To run a simple test, install the basic requirements and run:

```bash
bash test_one_slide.sh
```

It will:

* Downloads a slide from the TCGA-BRCA project (and put it under data_test)
* Run the `main.py` script on it (using gigassl_type=mlp and tile_encoder_type=moco)

It stores the extracted tiles and thumbnail under `./tmp`. Have a look at it!

Note: this test ran successfully in 13.3 sec on CPU using the exact environment described in ./tested_env.txt.

# Installation

## Basic requirements

We recommend using conda to install the necessary dependencies. If you don't have conda installed, please follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).
Then, in this repository, run:

```bash
conda env create -f requirements.yml && conda activate gigassl
```

With these requirements you will be able to run gigassl models of type `mlp` using the `moco` or `phikon` encoder.
They provide a quick way to test the package, as well as good WSIs embeddings (even if a bit less performant than the ones obtained with `scm` models).

## Requirements to run gigassl models of type `scm`

`scm` refers to SparseConvMIL-related models. These are the ones evaluated in the paper.

After having installed the basic requirements, you will need to install the [SparseConvNet](https://github.com/facebookresearch/SparseConvNet) library, following the instruction on its github repository.
In their setup instructions, do not forget to have at least one GPU available when launching `bash develop.sh`, otherwise the GPU version of the package will not be installed.

## Requirements to run gigassl trained on top of CTransPath embeddings

`scm` and `mlp` GigaSSL models have been trained on both MoCo (in-house model), Phikon and CTransPath embeddings.
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

|                   | GigaSSL + Moco | GigaSSL + CTransPath | GigaSSL + PhiKon |
|-------------------|----------------|----------------------|------------------|
| brca_histotypes (ductal / lobular)   | 0.927 ± 0.027  | 0.942 ± 0.029        | 0.946 ± 0.030    |
| brca_mhrd (HRD / HRP)        | 0.789 ± 0.027  | 0.804 ± 0.028        | 0.818 ± 0.024    |
| brca_moltype (TNBC / luminal)     | 0.929 ± 0.041  | 0.948 ± 0.035        | 0.941 ± 0.030    |
| kidney (cc / p / ch)           | 0.985 ± 0.007  | 0.994 ± 0.004        | 0.994 ± 0.006    |
| lung (LUAD / LUSC)             | 0.961 ± 0.013  | 0.974 ± 0.008        | 0.976 ± 0.008    |


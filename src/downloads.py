import os
from pathlib import Path
from huggingface_hub import snapshot_download
import gdown

import requests


download_params = {
    'ctranspath': {
        'filename': 'ctranspath.pth',
        'url': 'https://drive.google.com/uc?id=1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX',
        'use_gdown': True,
        'is_model': True
    },
    'phikon': {
        'filename': None,
        'url': 'hf-hub:owkin/phikon',
        'use_gdown': False,
        'is_model': True
    },
    'gigapath': {
        'filename': None,
        'url': 'hf_hub:prov-gigapath/prov-gigapath',
        'use_gdown': False,
        'is_model': True
    },
    'uni': {
        'filename': 'uni.pth',
        'url': '',
        'use_gdown': False,
        'is_model': True
    },
    'moco': {
        'filename': 'moco.pth.tar',
        'url': "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/8c93165a-87dd-4611-9147-5b4e4a38fd91/file_downloaded",
        'use_gdown': False,
        'is_model': True
    },
    'optimus': {
        'filename': None,
        'url': "hf_hub:bioptimus/H-optimus-0",
        'use_gdown': False,
        'is_model': True
    },
    'gigassl_scm_phikon': {
        'filename': 'gigassl-scm-phikon.pth.tar',
        'url': "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/80290ac0-8e39-4de3-a6a4-a1425c931ae6/file_downloaded",
        'use_gdown': False,
        'is_model': True
    },
    'gigassl_mlp_phikon': {
        'filename': 'gigassl-mlp-phikon.pth.tar',
        'url': "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/ac872e8f-9d6d-4b9d-b294-71205fb3aba7/file_downloaded",
        'use_gdown': False,
        'is_model': True
    },
    'gigassl_scm_moco': {
        'filename': 'gigassl-scm-moco.pth.tar',
        'url': "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/497532b6-a011-47a6-b831-fff5bae50bf6/file_downloaded",
        'use_gdown': False,
        'is_model': True
    },
    'gigassl_scm_ctranspath': {
        'filename': 'gigassl-scm-ctranspath.pth',
        'url': "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/703bfc34-e3f8-4305-bee0-e8a5b0424faf/file_downloaded",
        'use_gdown': False,
        'is_model': True
    },
    'gigassl_mlp_ctranspath': {
        'filename': 'gigassl-mlp-ctranspath.pth',
        'url': "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/5d788297-072a-44be-a889-62e32e4c7e67/file_downloaded",
        'use_gdown': False,
        'is_model': True
    },
    'gigassl_mlp_moco': {
        'filename': 'gigassl-mlp-moco.pth',
        'url': "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/b755938b-4edd-4139-8586-43b476b121b3/file_downloaded",
        'use_gdown': False,
        'is_model': True
    },
    'gigassl_scm_gigapath': {
        'filename': 'gigassl-scm-gigapath.pth.tar',
        'url': "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/8ebb6688-df30-40db-86da-524b83853def/file_downloaded",
        'use_gdown': False,
        'is_model': True
    },
    'gigassl_scm_uni': {
        'filename': 'gigassl-scm-uni.pth',
        'url': "",
        'use_gdown': False,
        'is_model': True
    },
    'gigassl_scm_optimus': {
        'filename': 'gigassl-scm-optimus.pth',
        'url': "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/e906e762-d542-4496-9814-9c35558d8fc8/file_downloaded",
        'use_gdown': False,
        'is_model': True
    },
    'TCGA_ctranspath': {
        'filename': 'TCGA-gigassl_ctranspath.npy',
        'url': "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/ccf402f7-34ee-4a74-8b8a-51d9e9cebea0/file_downloaded",
        'use_gdown': False,
        'is_model': False
    },
    'TCGA_moco': {
        'filename': 'TCGA-gigassl_moco.npy',
        'url': "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/2006d37e-61b0-469c-95f1-fda3df1766c4/file_downloaded",
        'use_gdown': False,
        'is_model': False
    },
    'TCGA_phikon': {
        'filename': 'TCGA-gigassl_phikon.npy',
        'url': "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/6ab7211d-f850-47b7-9091-54dc87abb66b/file_downloaded",
        'use_gdown': False,
        'is_model': False
    },
    'TCGA_optimus': {
        'filename': 'TCGA-gigassl_optimus.npy',
        'url': "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/400d7ad7-9e9f-48dd-b241-a054cc189a37/file_downloaded",
        'use_gdown': False,
        'is_model': False
    },
    'TCGA_gigapath': {
        'filename': 'TCGA-gigassl_gigapath.npy',
        'url':"https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/03977666-5ca2-43cc-a588-0f067dd1fc09/file_downloaded", 
        'use_gdown': False,
        'is_model': False
    },
    'pca_ctranspath': {
        'filename': 'pca-ctranspath.npy',
        'url': "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/b2301b46-433e-4028-aba1-853c71739638/file_downloaded",
        'use_gdown': False,
        'is_model': True
    },
    'pca_phikon': {
        'filename': 'pca-phikon.pth',
        'url': "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/2aa5a972-9b06-4fee-8ff0-f780b5f5b58e/file_downloaded",
        'use_gdown': False,
        'is_model': True
    },
    'pca_gigapath': {
        'filename': 'pca-gigapath.pth',
        'url': "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/4ff282b5-2031-4a33-987c-b73a6893d7d4/file_downloaded",
        'use_gdown': False,
        'is_model': True
    },
    'pca_uni': {
        'filename': 'pca-uni.pth',
        'url': "",
        'use_gdown': False,
        'is_model': True
    },
    'pca_optimus': {
        'filename': 'pca-optimus.pth',
        'url': "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/a8666183-bc01-48a6-9702-2f57e82735ad/file_downloaded",
        'use_gdown': False,
        'is_model': True
    }
}

def download_item(item_name):
    """Download a model or data item based on its name in the download_params dictionary."""
    if item_name not in download_params:
        print(f"Error: {item_name} not found in download parameters.")
        return None

    params = download_params[item_name]
    filename = params['filename']
    url = params['url']
    use_gdown = params['use_gdown']
    is_model = params['is_model']

    #HF download - using the HF cache
    if url.startswith('hf_hub:'):
        snapshot_download(url.split(':')[1])
        return url

    if filename is None:
        return None
    if is_model:
        return download_model(filename, url, use_gdown)
    else:
        return download_data(filename, url)

def download_from_url(url, path):
    """Download a file from a direct URL."""
    if path.exists():
        print(f'{str(path)} already exists')
        return str(path.resolve())
    else:
        print(f'Downloading {str(path)}')
        response = requests.get(url)
        response.raise_for_status() 
        with open(path, 'wb') as file:
            file.write(response.content)
        return str(path.resolve())

def download_model(filename, url="", use_gdown=False):
    model_root = Path('./models/')
    model_root.mkdir(exist_ok=True, parents=True)
    model_path = model_root / filename

    if url == '':
        return str(model_path.resolve())

    if use_gdown:
        if model_path.exists():
            print(f'{str(model_path)} already exists')
            return str(model_path.resolve())
        else:
            print(f'Downloading {str(model_path)}')
            gdown.download(url, str(model_path), quiet=False)
            return str(model_path.resolve())
    else:
        return download_from_url(url, model_path)

def download_data(filename, url):
    data_root = Path('./data_encoded')
    data_root.mkdir(exist_ok=True, parents=True)
    data_path = data_root / filename
    return download_from_url(url, data_path)

if __name__ == '__main__':
    # Test downloads
    for item_name in download_params:
        download_item(item_name)
    print('Done')

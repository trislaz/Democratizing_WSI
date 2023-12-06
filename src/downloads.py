import os
from pathlib import Path

import requests

def download_ctranspath():
    model_root = Path('./models/')
    if not model_root.exists():
        model_root.mkdir()
    model_path = model_root / 'ctranspath.pth'
    if model_path.exists():
        print(f'{str(model_path)} already exists')
        return str(model_path.resolve())
    else:
        print(f'Downloading {str(model_path)}')
        os.system(f"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX" -O {str(model_path.resolve())} && rm -rf /tmp/cookies.txtOD""")
        return str(model_path.resolve())

def download_moco():
    model_root = Path('./models/')
    if not model_root.exists():
        model_root.mkdir()
    model_path = model_root / 'moco.pth.tar'
    if model_path.exists():
        print(f'{str(model_path)} already exists')
        return str(model_path.resolve())
    else:
        print(f'Downloading {str(model_path)}')
        response = requests.get('https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/8c93165a-87dd-4611-9147-5b4e4a38fd91/file_downloaded')
        response.raise_for_status()  # Will raise an error for a bad status code
        with open(model_path, 'wb') as file:
            file.write(response.content)
        return str(model_path.resolve())

def download_gigassl_scm_moco():
    model_root = Path('./models/')
    if not model_root.exists():
        model_root.mkdir()
    model_path = model_root / 'gigassl-scm-moco.pth.tar'
    if model_path.exists():
        print(f'{str(model_path)} already exists')
        return str(model_path.resolve())
    else:
        print(f'Downloading {str(model_path)}')
        response = requests.get('https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/497532b6-a011-47a6-b831-fff5bae50bf6/file_downloaded')
        response.raise_for_status()  # Will raise an error for a bad status code
        with open(model_path, 'wb') as file:
            file.write(response.content)
        return str(model_path.resolve())

def download_gigassl_scm_ctranspath():
    model_root = Path('./models/')
    if not model_root.exists():
        model_root.mkdir()
    model_path = model_root / 'gigassl-scm-ctranspath.pth'
    if model_path.exists():
        print(f'{str(model_path)} already exists')
        return str(model_path.resolve())
    else:
        print(f'Downloading {str(model_path)}')
        response = requests.get('https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/703bfc34-e3f8-4305-bee0-e8a5b0424faf/file_downloaded')
        response.raise_for_status()  # Will raise an error for a bad status code
        with open(model_path, 'wb') as file:
            file.write(response.content)
        return str(model_path.resolve())

def download_gigassl_mlp_ctranspath():
    model_root = Path('./models/')
    if not model_root.exists():
        model_root.mkdir()
    model_path = model_root / 'gigassl-mlp-ctranspath.pth'
    if model_path.exists():
        print(f'{str(model_path)} already exists')
        return str(model_path.resolve())
    else:
        print(f'Downloading {str(model_path)}')
        response = requests.get('https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/5d788297-072a-44be-a889-62e32e4c7e67/file_downloaded')
        response.raise_for_status()  # Will raise an error for a bad status code
        with open(model_path, 'wb') as file:
            file.write(response.content)
        return str(model_path.resolve())

def download_gigassl_mlp_moco():
    model_root = Path('./models/')
    if not model_root.exists():
        model_root.mkdir()
    model_path = model_root / 'gigassl-mlp-moco.pth'
    if model_path.exists():
        print(f'{str(model_path)} already exists')
        return str(model_path.resolve())
    else:
        print(f'Downloading {str(model_path)}')
        response = requests.get('https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/b755938b-4edd-4139-8586-43b476b121b3/file_downloaded')
        response.raise_for_status()  # Will raise an error for a bad status code
        with open(model_path, 'wb') as file:
            file.write(response.content)
        return str(model_path.resolve())


def download_TCGA_ctranspath():
    data_root = Path('./data_encoded')
    if not data_root.exists():
        data_root.mkdir()
    data_path = data_root / 'TCGA-gigassl_ctranspath.npy'
    if data_path.exists():
        print(f'{str(data_path)} already exists')
        return str(data_path.resolve())
    else:
        print(f'Downloading {str(data_path)}')
        response = requests.get('https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/ccf402f7-34ee-4a74-8b8a-51d9e9cebea0/file_downloaded')
        response.raise_for_status()  # Will raise an error for a bad status code
        with open(data_path, 'wb') as file:
            file.write(response.content)
        return str(data_path.resolve())

def download_TCGA_moco():
    data_root = Path('./data_encoded')
    if not data_root.exists():
        data_root.mkdir()
    data_path = data_root / 'TCGA-gigassl_moco.npy'
    if data_path.exists():
        print(f'{str(data_path)} already exists')
        return str(data_path.resolve())
    else:
        print(f'Downloading {str(data_path)}')
        response = requests.get('https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/2006d37e-61b0-469c-95f1-fda3df1766c4/file_downloaded')
        response.raise_for_status()  # Will raise an error for a bad status code
        with open(data_path, 'wb') as file:
            file.write(response.content)
        return str(data_path.resolve())

def download_pca_ctranspath():
    model_root = Path('./models/')
    if not model_root.exists():
        model_root.mkdir()
    model_path = model_root / 'pca-ctranspath.npy'
    if model_path.exists():
        print(f'{str(model_path)} already exists')
        return str(model_path.resolve())
    else:
        print(f'Downloading {str(model_path)}')
        response = requests.get('https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/b2301b46-433e-4028-aba1-853c71739638/file_downloaded')
        response.raise_for_status()  # Will raise an error for a bad status code
        with open(model_path, 'wb') as file:
            file.write(response.content)
        return str(model_path.resolve())

if __name__ == '__main__':
    # Test dl
    download_ctranspath()
    download_moco()
    download_gigassl_scm_moco()
    download_gigassl_scm_ctranspath()
    download_gigassl_mlp_ctranspath()
    download_gigassl_mlp_moco()
    download_TCGA_ctranspath()
    download_TCGA_moco()
    download_pca_ctranspath()
    print('Done')
    

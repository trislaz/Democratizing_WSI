import os
from pathlib import Path

import requests

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

def download_phikon():
    return None

def download_moco():
    url = "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/8c93165a-87dd-4611-9147-5b4e4a38fd91/file_downloaded"
    model_root = Path('./models/')
    model_root.mkdir(exist_ok=True, parents=True)
    model_path = model_root / 'moco.pth.tar'
    return download_from_url(url, model_path)

def download_gigassl_scm_phikon():
    url = "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/80290ac0-8e39-4de3-a6a4-a1425c931ae6/file_downloaded"
    model_root = Path('./models/')
    model_root.mkdir(exist_ok=True, parents=True)
    model_path = model_root / 'gigassl-scm-phikon.pth.tar'
    return download_from_url(url, model_path)

def download_gigassl_mlp_phikon():
    url = "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/ac872e8f-9d6d-4b9d-b294-71205fb3aba7/file_downloaded"
    model_root = Path('./models/')
    model_root.mkdir(exist_ok=True, parents=True)
    model_path = model_root / 'gigassl-mlp-phikon.pth.tar'
    return download_from_url(url, model_path)

def download_gigassl_scm_moco():
    url = "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/497532b6-a011-47a6-b831-fff5bae50bf6/file_downloaded"
    model_root = Path('./models/')
    model_root.mkdir(exist_ok=True, parents=True)
    model_path = model_root / 'gigassl-scm-moco.pth.tar'
    return download_from_url(url, model_path)

def download_gigassl_scm_ctranspath():
    url = "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/703bfc34-e3f8-4305-bee0-e8a5b0424faf/file_downloaded"
    model_root = Path('./models/')
    model_root.mkdir(exist_ok=True, parents=True)
    model_path = model_root / 'gigassl-scm-ctranspath.pth'
    return download_from_url(url, model_path)

def download_gigassl_mlp_ctranspath():
    url = "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/5d788297-072a-44be-a889-62e32e4c7e67/file_downloaded"
    model_root = Path('./models/')
    model_root.mkdir(exist_ok=True, parents=True)
    model_path = model_root / 'gigassl-mlp-ctranspath.pth'
    return download_from_url(url, model_path)   

def download_gigassl_mlp_moco():
    url = "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/b755938b-4edd-4139-8586-43b476b121b3/file_downloaded"
    model_root = Path('./models/')
    model_root.mkdir(exist_ok=True, parents=True)
    model_path = model_root / 'gigassl-mlp-moco.pth'
    return download_from_url(url, model_path)

def download_TCGA_ctranspath():
    url = "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/ccf402f7-34ee-4a74-8b8a-51d9e9cebea0/file_downloaded"
    data_root = Path('./data_encoded')
    data_root.mkdir(exist_ok=True, parents=True)
    data_path = data_root / 'TCGA-gigassl_ctranspath.npy'
    return download_from_url(url, data_path)

def download_TCGA_moco():
    url = "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/2006d37e-61b0-469c-95f1-fda3df1766c4/file_downloaded"
    data_root = Path('./data_encoded')
    data_root.mkdir(exist_ok=True, parents=True)
    data_path = data_root / 'TCGA-gigassl_moco.npy'
    return download_from_url(url, data_path)

def download_pca_ctranspath():
    url = "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/b2301b46-433e-4028-aba1-853c71739638/file_downloaded"
    model_root = Path('./models/')
    model_root.mkdir(exist_ok=True, parents=True)
    model_path = model_root / 'pca-ctranspath.npy'
    return download_from_url(url, model_path)

def download_pca_phikon():
    url = "https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/2aa5a972-9b06-4fee-8ff0-f780b5f5b58e/file_downloaded"
    model_root = Path('./models/')
    model_root.mkdir(exist_ok=True, parents=True)
    model_path = model_root / 'pca-phikon.pth'
    return download_from_url(url, model_path)

if __name__ == '__main__':
    # Test dl
    download_ctranspath()
    download_moco()
    download_gigassl_scm_moco()
    download_gigassl_scm_ctranspath()
    download_gigassl_mlp_ctranspath()
    download_gigassl_mlp_moco()
    download_gigassl_scm_phikon()
    download_gigassl_mlp_phikon()
    download_pca_phikon()
    download_TCGA_ctranspath()
    download_TCGA_moco()
    download_pca_ctranspath()
    download_pca_phikon() 
    print('Done')
    

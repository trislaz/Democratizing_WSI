"""Encode WSI into a feature vector."""
import argparse
import os
from pathlib import Path
import numpy as np
from filelock import Timeout, FileLock


parser = argparse.ArgumentParser(description='Encode WSI into a feature vector.')
parser.add_argument('--input', type=str, help='Path to the input WSIs. Can be a directory or a single file.')
parser.add_argument('--output', type=str, help='Path to write the output feature vectors.')
parser.add_argument('--gigassl_type', type=str, help='name of the gigassl type - SparseConvMIL or MLP-like-', default='scm', choices=['scm', 'mlp'])
parser.add_argument('--tile_encoder_type', type=str, help='name of the tile encoder type', default='moco', choices=['moco', 'ctranspath', 'phikon', 'gigapath', 'uni', 'optimus'])
parser.add_argument('--N_ensemble', type=int, default=100, help='Number of WSI views to ensemble (the higher the better but the slower).')
parser.add_argument('--store_intermediate', type=str, default=None, help='Path to store the tiles.')
parser.add_argument('--job_id', type=int, default=None, help='Job id -if used with a cluster-. Do not use it if you are running the script locally.')
args = parser.parse_args()

output = Path(args.output)
output.mkdir(exist_ok=True, parents=True)

# Useful for imports
os.environ['ENCODER_TYPE'] = args.tile_encoder_type.upper()
os.environ['ARCHITECTURE'] = args.gigassl_type.upper()
from src.encode_slide import encode_image

if args.job_id is not None:
    from_to = [args.job_id*500, (args.job_id+1)*500]
else:
    from_to =[0, -1] 

output_file = output / f'features_{Path(args.input).stem}_{Path(args.tile_encoder_type).stem}_{args.gigassl_type}.npy'
lock_file = output_file.with_suffix('.lock')

dico = encode_image(tile_encoder_type=args.tile_encoder_type, gigassl_type=args.gigassl_type, image_path=args.input, N_ensemble=args.N_ensemble, store_intermediate=args.store_intermediate, from_to=from_to)

with FileLock(lock_file, timeout=10):
    if output_file.exists():
        existing_data = np.load(output_file, allow_pickle=True).item()
        dico.update(existing_data)
    np.save(output_file, dico)

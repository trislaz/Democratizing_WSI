#!/bin/bash
#SBATCH --array=0-100%5
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=1             # Nombre total de processus MPI
#SBATCH --ntasks-per-node=1    # Nombre de processus MPI par noeud
# Dans le vocabulaire Slurm "multithread" fait référence à l"hyperthreading.
#SBATCH --hint=nomultithread   # 1 processus MPI par coeur physique (pas d"hyperthreading)
#SBATCH --output=logs/%x_%a.out  # Nom du fichier de sortie contenant l"ID et l"indice

slides='/path/to/slide/folder/'
output='./outs/'
gigassl_type='mlp'
tile_encoder_type='phikon'
N_ensemble=50
job_id=$SLURM_ARRAY_TASK_ID

python main.py --input $slides --output $output --gigassl_type $gigassl_type --tile_encoder_type $tile_encoder_type --N_ensemble $N_ensemble --job_id $job_id


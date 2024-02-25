#!/bin/bash
#SBATCH --array=0-100%5
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=1             
#SBATCH --ntasks-per-node=1    
#SBATCH --hint=nomultithread   
#SBATCH --output=logs/%x_%a.out 

slides='/path/to/slide/folder/'
output='./outs/'
gigassl_type='mlp'
tile_encoder_type='phikon'
N_ensemble=50
job_id=$SLURM_ARRAY_TASK_ID

python main.py --input $slides --output $output --gigassl_type $gigassl_type --tile_encoder_type $tile_encoder_type --N_ensemble $N_ensemble --job_id $job_id


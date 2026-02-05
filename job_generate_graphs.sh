#!/bin/bash
#SBATCH --job-name=nb_graph_gen
#SBATCH --account=fc_neuronident
#SBATCH --partition=savio3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20        
#SBATCH --time=24:00:00           
#SBATCH --o nb_graphs_%j.out

# Cargar entorno
source /global/home/users/$USER/scratch/neuroback/.venv/bin/activate

echo "Iniciando generación de grafos..."
python3 graph.py pretrain
python3 graph.py finetune
python3 graph.py validation

echo "Proceso finalizado."
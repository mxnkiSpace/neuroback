#!/bin/bash
#SBATCH --job-name=nb_graph_array
#SBATCH --account=fc_neuronident
#SBATCH --partition=savio3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20        
#SBATCH --time=04:00:00           
#SBATCH -o logs/nb_%j_%a.out

# ==============================================================================
# USO: sbatch --array=0-N job_generate_graphs.sh <pretrain|finetune|validation>
# Ejemplo: sbatch --array=0-23 job_generate_graphs.sh pretrain
# ==============================================================================

SET_TYPE=$1  # pretrain, finetune o validation
SCRATCH_BASE="/global/home/users/$USER/scratch/neuroback/data"
SCRATCH_INPUT="$SCRATCH_BASE/cnf/$SET_TYPE"
SCRATCH_BB="$SCRATCH_BASE/backbone/$SET_TYPE"
SCRATCH_OUTPUT="$SCRATCH_BASE/pt/$SET_TYPE"

# Carpeta temporal única por tarea de la array en el disco local del nodo
LOCAL_TMP="/tmp/nb_${SET_TYPE}_${SLURM_ARRAY_TASK_ID}"
BATCH_FILE="batches/${SET_TYPE}/batch_$(printf "%02d" $SLURM_ARRAY_TASK_ID).txt"

# Validamos
if [ ! -f "$BATCH_FILE" ]; then
    echo "Error: Batch $BATCH_FILE no encontrado. Asegúrate de correr make_batches.sh primero."
    exit 1
fi

# Preparamos entorno local en el nodo 
echo "Iniciando tarea $SLURM_ARRAY_TASK_ID para $SET_TYPE..."
mkdir -p $LOCAL_TMP/input $LOCAL_TMP/output $LOCAL_TMP/backbone

# Copiamos archivos del batch desde el DFS al nodo local
echo "Copiando archivos al nodo local..."
while read file; do
    # Copiar CNF
    cp "$SCRATCH_INPUT/$file" "$LOCAL_TMP/input/"
    
    # Copiar Backbone correspondiente
    bb_file="${file}.backbone.xz"
    if [ -f "$SCRATCH_BB/$bb_file" ]; then
        cp "$SCRATCH_BB/$bb_file" "$LOCAL_TMP/backbone/"
    else
        # Intento con el nombre alternativo
        bb_alt=$(echo $file | sed 's/\.xz$//').backbone.xz
        [ -f "$SCRATCH_BB/$bb_alt" ] && cp "$SCRATCH_BB/$bb_alt" "$LOCAL_TMP/backbone/"
    fi
done < $BATCH_FILE

# Ejecutamos procesamiento de grafos
source /global/home/users/$USER/scratch/neuroback/.venv/bin/activate
python3 graph.py $SET_TYPE $LOCAL_TMP/input $LOCAL_TMP/output $BATCH_FILE $LOCAL_TMP/backbone

# Sincronizamos resultados de vuelta al almacenamiento permanente
echo "Sincronizando resultados a Scratch..."
mkdir -p $SCRATCH_OUTPUT/processed
rsync -av $LOCAL_TMP/output/processed/ $SCRATCH_OUTPUT/processed/

# Limpiamos del nodo local
echo "Limpiando nodo local..."
rm -rf $LOCAL_TMP

echo "Trabajo finalizado para el batch $(basename $BATCH_FILE)."
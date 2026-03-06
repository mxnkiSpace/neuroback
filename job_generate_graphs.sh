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
# ==============================================================================

SET_TYPE=$1

## Directorio base
BASE_DIR="/global/home/users/$USER/scratch/neuroback/neuroback"
DATA_DIR="$BASE_DIR/data"
BATCH_FILE="$BASE_DIR/batches/${SET_TYPE}/batch_$(printf "%02d" $SLURM_ARRAY_TASK_ID).txt"

# Memoria temporal
LOCAL_TMP="/tmp/nb_${SET_TYPE}_${SLURM_ARRAY_TASK_ID}"

if [ ! -f "$BATCH_FILE" ]; then
    echo "Error: Batch $BATCH_FILE no encontrado. Finalizando tarea."
    exit 0
fi

echo "Iniciando procesamiento del batch: $(basename $BATCH_FILE)"

mkdir -p $LOCAL_TMP/input $LOCAL_TMP/output $LOCAL_TMP/backbone

# local son los archivos que vienen del repo original (datos de prueba)
grep "^orig" $BATCH_FILE | awk '{print $2}' > $LOCAL_TMP/orig_cnf_list.txt
grep "^dual" $BATCH_FILE | awk '{print $2}' > $LOCAL_TMP/dual_cnf_list.txt
grep "^local" $BATCH_FILE | awk '{print $2}' > $LOCAL_TMP/local_cnf_list.txt

# Generamos los nombres para los archivos bb
sed 's/$/.backbone.xz/' $LOCAL_TMP/orig_cnf_list.txt > $LOCAL_TMP/orig_bb_list.txt
sed 's/$/.backbone.xz/' $LOCAL_TMP/dual_cnf_list.txt > $LOCAL_TMP/dual_bb_list.txt
sed 's/$/.backbone.xz/' $LOCAL_TMP/local_cnf_list.txt > $LOCAL_TMP/local_bb_list.txt

# Determinar el prefijo del tarball (pt o ft)
if [ "$SET_TYPE" == "pretrain" ]; then PREFIX="pt"; else PREFIX="ft"; fi

# Se extrae solo lo que se necesita
echo "Extrayendo archivos originales..."
if [ -s $LOCAL_TMP/orig_cnf_list.txt ]; then
    tar -xzf "$DATA_DIR/cnf/$SET_TYPE/cnf_${PREFIX}.tar.gz" -C "$LOCAL_TMP/input/" -T $LOCAL_TMP/orig_cnf_list.txt
    tar -xzf "$DATA_DIR/backbone/$SET_TYPE/bb_${PREFIX}.tar.gz" -C "$LOCAL_TMP/backbone/" -T $LOCAL_TMP/orig_bb_list.txt
fi

echo "Extrayendo archivos duales..."
if [ -s $LOCAL_TMP/dual_cnf_list.txt ]; then
    tar -xzf "$DATA_DIR/cnf/$SET_TYPE/d_cnf_${PREFIX}.tar.gz" -C "$LOCAL_TMP/input/" -T $LOCAL_TMP/dual_cnf_list.txt
    tar -xzf "$DATA_DIR/backbone/$SET_TYPE/d_bb_${PREFIX}.tar.gz" -C "$LOCAL_TMP/backbone/" -T $LOCAL_TMP/dual_bb_list.txt
fi

echo "Copiando archivos locales (validation / sueltos)..."
if [ -s $LOCAL_TMP/local_cnf_list.txt ]; then
    while read file; do
        cp "$DATA_DIR/cnf/$SET_TYPE/$file" "$LOCAL_TMP/input/"
        if [ -f "$DATA_DIR/backbone/$SET_TYPE/${file}.backbone.xz" ]; then
            cp "$DATA_DIR/backbone/$SET_TYPE/${file}.backbone.xz" "$LOCAL_TMP/backbone/"
        fi
    done < $LOCAL_TMP/local_cnf_list.txt
fi

# Limpiamos la lista del batch
awk '{print $2}' $BATCH_FILE > $LOCAL_TMP/clean_batch.txt

# Ejecutamos graph.py
echo "Procesando grafos con graph.py..."
source $BASE_DIR/.venv/bin/activate
python3 $BASE_DIR/graph.py $SET_TYPE $LOCAL_TMP/input $LOCAL_TMP/output $LOCAL_TMP/clean_batch.txt $LOCAL_TMP/backbone

# Guardamos resultados
echo "Guardando resultados en Scratch y limpiando nodo..."
mkdir -p "$DATA_DIR/pt/$SET_TYPE/processed"
rsync -a $LOCAL_TMP/output/processed/ "$DATA_DIR/pt/$SET_TYPE/processed/"
rm -rf $LOCAL_TMP

echo "#=== Tarea completada ===#"
#!/bin/bash
#SBATCH --job-name=nb_learn
#SBATCH --account=fc_neuronident
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:GTX2080TI:1
#SBATCH --time=24:00:00
#SBATCH -o logs/nb_learn_%j.out

# ==============================================================================
# USO (local):   bash job_learn.sh <pretrain|finetune>
# USO (cluster): sbatch job_learn.sh <pretrain|finetune>
# ==============================================================================

SET_TYPE=$1

# Validación de argumento
if [ -z "$SET_TYPE" ]; then
    echo "Error: Debe especificar <pretrain|finetune>"
    exit 1
fi
if [ "$SET_TYPE" != "pretrain" ] && [ "$SET_TYPE" != "finetune" ]; then
    echo "Error: <set_type> debe ser 'pretrain' o 'finetune', recibido '$SET_TYPE'"
    exit 1
fi

# ==============================================================================
# DETECCIÓN DE ENTORNO (local vs SAVIO)
# ==============================================================================
if [ -d "/global/home/users/$USER/scratch" ]; then
    BASE_DIR="/global/home/users/$USER/scratch/neuroback/neuroback"
    RUNNING_ENV="savio"
    SRC_TRAIN_DIR="$BASE_DIR/data/pt/$SET_TYPE/processed"
    SRC_VLD_DIR="$BASE_DIR/data/pt/validation/processed"
else
    BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    RUNNING_ENV="local"
    SRC_TRAIN_DIR="$BASE_DIR/data/processed/$SET_TYPE"
    SRC_VLD_DIR="$BASE_DIR/data/processed/validation"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Entorno=$RUNNING_ENV | BASE_DIR=$BASE_DIR | SET_TYPE=$SET_TYPE"

# ==============================================================================
# VALIDACIONES PREVIAS
# ==============================================================================
if [ ! -d "$SRC_TRAIN_DIR" ]; then
    echo "Error: No existe directorio de entrenamiento: $SRC_TRAIN_DIR"
    exit 1
fi
if [ ! -d "$SRC_VLD_DIR" ]; then
    echo "Error: No existe directorio de validación: $SRC_VLD_DIR"
    exit 1
fi

# Finetune requiere checkpoint del pretrain
if [ "$SET_TYPE" == "finetune" ] && [ ! -f "$BASE_DIR/models/pretrain/pretrain-best.ptg" ]; then
    echo "Error: finetune requiere $BASE_DIR/models/pretrain/pretrain-best.ptg"
    echo "       Corré primero: $0 pretrain"
    exit 1
fi

# ==============================================================================
# WORKING DIRECTORY EN DISCO LOCAL DEL NODO
# ==============================================================================
JOB_ID=${SLURM_JOB_ID:-local}
WORK_DIR="/tmp/nb_learn_${SET_TYPE}_${JOB_ID}"
echo "[$(date '+%H:%M:%S')] Working dir: $WORK_DIR"

# Limpieza preventiva por si quedó basura de un job previo
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR/data/pt/$SET_TYPE/processed"
mkdir -p "$WORK_DIR/data/pt/validation/processed"

# ==============================================================================
# FUNCIÓN: Poblar directorio processed/ desde fuente
#   - Si hay *.tar.gz: descomprimirlos
#   - Si hay *.pt sueltos: copiarlos
# ==============================================================================
populate_processed() {
    local src=$1
    local dest=$2
    local label=$3

    local tar_count=$(find "$src" -maxdepth 1 -name '*.tar.gz' | wc -l)
    local pt_count=$(find "$src" -maxdepth 1 -name '*.pt' | wc -l)

    if [ "$tar_count" -gt 0 ]; then
        echo "[$(date '+%H:%M:%S')] $label: descomprimiendo $tar_count tarball(s) desde $src"
        for tar in "$src"/*.tar.gz; do
            echo "  - $(basename "$tar")"
            tar -xzf "$tar" -C "$dest/" || { echo "Error extrayendo $tar"; return 1; }
        done
    elif [ "$pt_count" -gt 0 ]; then
        echo "[$(date '+%H:%M:%S')] $label: copiando $pt_count archivos .pt desde $src"
        cp "$src"/*.pt "$dest/" || return 1
    else
        echo "Error: $label no tiene ni *.tar.gz ni *.pt en $src"
        return 1
    fi

    local total=$(ls "$dest" | wc -l)
    local size=$(du -sh "$dest" | cut -f1)
    echo "[$(date '+%H:%M:%S')] $label: $total archivos en destino ($size)"
    return 0
}

echo "[$(date '+%H:%M:%S')] PASO 1: Poblando datasets..."
populate_processed "$SRC_TRAIN_DIR" "$WORK_DIR/data/pt/$SET_TYPE/processed" "train ($SET_TYPE)" || exit 1
populate_processed "$SRC_VLD_DIR"   "$WORK_DIR/data/pt/validation/processed" "validation"        || exit 1

# ==============================================================================
# SYMLINKS PARA QUE models/ Y log/ PERSISTAN FUERA DEL NODO
# learn.py escribe a ./models/{pretrain,finetune}/*.ptg y ./log/{pretrain,finetune}/*.log
# (rutas relativas al cwd que será WORK_DIR)
# ==============================================================================
mkdir -p "$BASE_DIR/models" "$BASE_DIR/log"
ln -s "$BASE_DIR/models" "$WORK_DIR/models"
ln -s "$BASE_DIR/log"    "$WORK_DIR/log"

# ==============================================================================
# EJECUTAR learn.py
# ==============================================================================
echo "[$(date '+%H:%M:%S')] PASO 2: Ejecutando learn.py $SET_TYPE..."

# Activar virtualenv si existe
if [ -f "$BASE_DIR/.venv/bin/activate" ]; then
    source "$BASE_DIR/.venv/bin/activate"
elif [ -f "$BASE_DIR/venv/bin/activate" ]; then
    source "$BASE_DIR/venv/bin/activate"
fi

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 no encontrado"
    exit 1
fi

cd "$WORK_DIR" || exit 1
# Usamos _run_learn.py (wrapper) para forzar multiprocessing start_method='fork'
# en Python 3.14+ sin tener que editar learn.py.
python3 "$BASE_DIR/_run_learn.py" "$SET_TYPE"
RET=$?

if [ $RET -ne 0 ]; then
    echo "Error: learn.py salió con código $RET"
    echo "NO eliminando $WORK_DIR para debugging"
    exit $RET
fi

# ==============================================================================
# LIMPIEZA — models/ y log/ son symlinks, sus targets persisten
# ==============================================================================
echo "[$(date '+%H:%M:%S')] PASO 3: Limpiando $WORK_DIR..."
cd "$BASE_DIR"
rm -rf "$WORK_DIR"

echo "[$(date '+%H:%M:%S')] ✓ Training completado ($SET_TYPE)"

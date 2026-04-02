#!/bin/bash
#SBATCH --job-name=nb_graph_array
#SBATCH --account=fc_neuronident
#SBATCH --partition=savio3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10        
#SBATCH --time=04:00:00           
#SBATCH -o logs/nb_%j_%a.out

# ==============================================================================
# USO (local):   bash job_generate_graphs.sh <pretrain|finetune|validation> <batch_id>
# USO (cluster): sbatch --array=0-N job_generate_graphs.sh <pretrain|finetune|validation>
# ==============================================================================

SET_TYPE=$1
BATCH_ID=$2

# Validación: Debe especificar SET_TYPE
if [ -z "$SET_TYPE" ]; then
    echo "Error: Debe especificar <pretrain|finetune|validation>"
    exit 1
fi

# Validar que no falten ambos y que no existan ambos a la vez
if [ -z "$BATCH_ID" ] && [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: Debe especificar <batch_id> o ejecutar con sbatch --array"
    exit 1
elif [ -n "$BATCH_ID" ] && [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: No se puede usar --array y <batch_id> simultáneamente"
    exit 1
fi

# Usar BATCH_ID en modo local, SLURM_ARRAY_TASK_ID en cluster
if [ -n "$BATCH_ID" ]; then
    SLURM_ARRAY_TASK_ID="$BATCH_ID"
fi

# ==============================================================================
# DETECCIÓN DE ENTORNO (local vs SAVIO)
# ==============================================================================
if [ -d "/global/home/users/$USER/scratch" ]; then
    # SAVIO Cluster
    BASE_DIR="/global/home/users/$USER/scratch/neuroback/neuroback"
    RUNNING_ENV="savio"
else
    # Local - usar ruta relativa
    BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    RUNNING_ENV="local"
fi

DATA_DIR="$BASE_DIR/data"
BATCH_FILE="$BASE_DIR/batches/${SET_TYPE}/batch_$(printf "%02d" $SLURM_ARRAY_TASK_ID).txt"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Entorno: $RUNNING_ENV | BASE_DIR: $BASE_DIR"

# Validación del batch
if [ ! -f "$BATCH_FILE" ]; then
    echo "Error: Batch $BATCH_FILE no encontrado."
    exit 1
fi

# Carpeta temporal en el disco local del nodo
LOCAL_TMP="/tmp/nb_${SET_TYPE}_${SLURM_ARRAY_TASK_ID}"

# Directorio donde se descomprimirán los tars (dentro del disco local del nodo para mejor rendimiento)
EXTRACTED_DIR="$LOCAL_TMP/extracted"

echo "Iniciando procesamiento del batch: $(basename $BATCH_FILE)"
echo "Archivos extraídos en: $EXTRACTED_DIR"
echo "Carpeta temporal de trabajo: $LOCAL_TMP"

# ==============================================================================
# FUNCIÓN: Descomprimir todos los tars
# ==============================================================================
decompress_all_tars() {
    echo "[$(date '+%H:%M:%S')] Descomprimiendo archivos..."
    mkdir -p "$EXTRACTED_DIR"
    
    # Descomprimir CNF originales
    local cnf_tar="$DATA_DIR/cnf/$SET_TYPE/cnf_${PREFIX}.tar.gz"
    if [ -f "$cnf_tar" ]; then
        echo "  - Descomprimiendo: cnf_${PREFIX}.tar.gz (archivos originales)"
        tar -xzf "$cnf_tar" -C "$EXTRACTED_DIR/"
    fi
    
    # Descomprimir CNF duales
    local cnf_dual_tar="$DATA_DIR/cnf/$SET_TYPE/d_cnf_${PREFIX}.tar.gz"
    if [ -f "$cnf_dual_tar" ]; then
        echo "  - Descomprimiendo: d_cnf_${PREFIX}.tar.gz (archivos duales)"
        tar -xzf "$cnf_dual_tar" -C "$EXTRACTED_DIR/"
    fi
    
    # Descomprimir Backbones originales
    local bb_tar="$DATA_DIR/backbone/$SET_TYPE/bb_${PREFIX}.tar.gz"
    if [ -f "$bb_tar" ]; then
        echo "  - Descomprimiendo: bb_${PREFIX}.tar.gz (backbones originales)"
        tar -xzf "$bb_tar" -C "$EXTRACTED_DIR/"
    fi
    
    # Descomprimir Backbones duales
    local bb_dual_tar="$DATA_DIR/backbone/$SET_TYPE/d_bb_${PREFIX}.tar.gz"
    if [ -f "$bb_dual_tar" ]; then
        echo "  - Descomprimiendo: d_bb_${PREFIX}.tar.gz (backbones duales)"
        tar -xzf "$bb_dual_tar" -C "$EXTRACTED_DIR/"
    fi
    
    echo "[$(date '+%H:%M:%S')] ✓ Descompresión completada"
    return 0
}

# ==============================================================================
# FUNCIÓN: Copiar archivo desde directorio extraído o archivos sueltos
# ==============================================================================
copy_file() {
    local filename=$1
    local dest_dir=$2
    local dir_type=$3  # "cnf" o "bb" (después de descomprimir)
    
    # Búsqueda 1: Archivos sueltos originales
    if [ -f "$DATA_DIR/${dir_type}/$SET_TYPE/$filename" ]; then
        cp "$DATA_DIR/${dir_type}/$SET_TYPE/$filename" "$dest_dir/"
        return 0
    fi
    
    # Búsqueda 2: Archivos descomprimidos en directorio extraído
    # Formato: cnf_pt/filename o bb_pt/filename
    local extracted_path="$EXTRACTED_DIR/${dir_type}_${PREFIX}/$filename"
    if [ -f "$extracted_path" ]; then
        cp "$extracted_path" "$dest_dir/"
        return 0
    fi
    
    # Búsqueda 3: Con prefijo d_ en archivos duales
    # Formato: d_cnf_pt/d_filename o d_bb_pt/d_filename
    # Detectar si el archivo ya tiene prefijo d_ para evitar d_d_
    if [[ "$filename" == d_* ]]; then
        # Filename ya tiene prefijo d_, usar tal cual
        local extracted_path_d="$EXTRACTED_DIR/d_${dir_type}_${PREFIX}/$filename"
    else
        # Agregar prefijo d_
        local filename_with_d="d_$filename"
        local extracted_path_d="$EXTRACTED_DIR/d_${dir_type}_${PREFIX}/$filename_with_d"
    fi
    if [ -f "$extracted_path_d" ]; then
        cp "$extracted_path_d" "$dest_dir/$filename"
        return 0
    fi
    
    return 1
}

# Preparamos entorno local en el nodo
mkdir -p "$LOCAL_TMP/input" "$LOCAL_TMP/output" "$LOCAL_TMP/backbone"

# ==============================================================================
# FUNCIÓN: Buscar archivo backbone con fallbacks
# Args: $1=cnf_filename, $2=destino, $3=prefix
# ==============================================================================
find_and_copy_backbone() {
    local cnf_name=$1
    local dest_dir=$2
    local prefix=$3
    
    # Variante 1: cnf_name.backbone.xz 
    # Ej: vlsat_49200_7490695.mcc2020_cnf.bz2 → vlsat_49200_7490695.mcc2020_cnf.backbone.xz
    local bb_name=$(echo "$cnf_name" | sed 's/\.[a-z0-9]*$//')
    bb_name="${bb_name}.backbone.xz"
    
    if copy_file "$bb_name" "$dest_dir" "bb"; then
        return 0
    fi
    
    # No es crítico si no existe backbone
    return 1
}

# ==============================================================================
# PROCESAR BATCH
# ==============================================================================
if [ "$SET_TYPE" == "pretrain" ]; then 
    PREFIX="pt"
else 
    PREFIX="ft"
fi

# PASO 1: Descomprimir todos los tars (una sola vez por SET_TYPE)
echo "[$(date '+%H:%M:%S')] PASO 1: Descomprimiendo archivos..."
decompress_all_tars
if [ $? -ne 0 ]; then
    echo "Error: No se pudo descomprimir archivos"
    exit 1
fi

# PASO 2: Procesar batch
echo "[$(date '+%H:%M:%S')] PASO 2: Procesando archivos del batch (PREFIX=$PREFIX)..."

processed_count=0
failed_count=0
while IFS= read -r cnf_filename; do
    # Trimear espacios y caracteres especiales
    cnf_filename=$(echo "$cnf_filename" | sed 's/^[[:space:]~]*//' | sed 's/[[:space:]]*$//')
    
    # Ignorar líneas vacías
    [ -z "$cnf_filename" ] && continue
    
    # Copiar CNF
    if copy_file "$cnf_filename" "$LOCAL_TMP/input" "cnf"; then
        echo "  ✓ $cnf_filename"
        ((processed_count++))
    else
        echo "  ✗ $cnf_filename (NO ENCONTRADO)"
        ((failed_count++))
        continue
    fi
    
    # Copiar backbone si existe
    find_and_copy_backbone "$cnf_filename" "$LOCAL_TMP/backbone" "$PREFIX"
    
done < "$BATCH_FILE"

echo "[$(date '+%H:%M:%S')] Resumen: $processed_count archivos procesados, $failed_count errores"

if [ $processed_count -eq 0 ]; then
    echo "Error: No se procesó ningún archivo"
    exit 1
fi

# Crear archivo limpio para python
cat "$BATCH_FILE" | grep -v '^$' > "$LOCAL_TMP/clean_batch.txt"

# Pre-creamos subcarpetas para evitar errores de multiprocessing en python
mkdir -p "$LOCAL_TMP/output/processed"

# ==============================================================================
# EJECUCIÓN DE graph.py
# ==============================================================================
echo "[$(date '+%H:%M:%S')] PASO 3: Ejecutando graph.py..."

# Detectar y activar virtualenv si existe
if [ -f "$BASE_DIR/.venv/bin/activate" ]; then
    source "$BASE_DIR/.venv/bin/activate"
elif [ -f "$BASE_DIR/venv/bin/activate" ]; then
    source "$BASE_DIR/venv/bin/activate"
fi

if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 no encontrado"
    exit 1
fi

python3 "$BASE_DIR/graph.py" "$SET_TYPE" "$LOCAL_TMP/input" "$LOCAL_TMP/output" "$LOCAL_TMP/clean_batch.txt" "$LOCAL_TMP/backbone"

if [ $? -ne 0 ]; then
    echo "Error: graph.py falló"
    # NO eliminar $LOCAL_TMP para debugging
    exit 1
fi

# ==============================================================================
# COPIAR RESULTADOS (comprimidos)
# ==============================================================================
echo "[$(date '+%H:%M:%S')] PASO 4: Guardando resultados..."

# Ubicación destino depende del entorno
if [ "$RUNNING_ENV" == "savio" ]; then
    RESULTS_DIR="$DATA_DIR/pt/$SET_TYPE/processed"
else
    # En local, guardar en data/
    RESULTS_DIR="$DATA_DIR/processed/$SET_TYPE"
fi

mkdir -p "$RESULTS_DIR"

if [ -d "$LOCAL_TMP/output/processed" ]; then
    # Comprimir resultados en archivo tar.gz
    echo "[$(date '+%H:%M:%S')] Comprimiendo resultados..."
    RESULTS_ARCHIVE="$LOCAL_TMP/results_batch_${SLURM_ARRAY_TASK_ID}.tar.gz"
    tar -czf "$RESULTS_ARCHIVE" -C "$LOCAL_TMP/output/processed" . 2>/dev/null
    
    # Copiar solo el archivo comprimido
    echo "[$(date '+%H:%M:%S')] Copiando archivo comprimido..."
    cp "$RESULTS_ARCHIVE" "$RESULTS_DIR/"
    
    # Descomprimir en destino
    echo "[$(date '+%H:%M:%S')] Descomprimiendo en destino..."
    tar -xzf "$RESULTS_DIR/$(basename $RESULTS_ARCHIVE)" -C "$RESULTS_DIR/" 2>/dev/null
    
    # Eliminar el archivo comprimido del destino
    rm "$RESULTS_DIR/$(basename $RESULTS_ARCHIVE)"
    
    echo "Resultados guardados en: $RESULTS_DIR"
else
    echo "Advertencia: No hay resultados en $LOCAL_TMP/output/processed"
fi

# ==============================================================================
# LIMPIEZA
# ==============================================================================
echo "[$(date '+%H:%M:%S')] PASO 5: Limpiando directorios temporales..."
rm -rf "$LOCAL_TMP"

echo "[$(date '+%H:%M:%S')] ✓ Tarea completada exitosamente"
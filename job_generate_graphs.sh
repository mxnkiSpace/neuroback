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

echo "Iniciando procesamiento del batch: $(basename $BATCH_FILE)"
echo "Carpeta temporal: $LOCAL_TMP"

# Preparamos entorno local en el nodo
mkdir -p "$LOCAL_TMP/input" "$LOCAL_TMP/output" "$LOCAL_TMP/backbone"

# ==============================================================================
# FUNCIÓN: Buscar y extraer archivo de .tar.gz
# Args: $1=archivo_a_buscar, $2=directorio_destino, $3=prefijo_tar (pt/ft)
# ==============================================================================
extract_from_tar() {
    local filename=$1
    local dest_dir=$2
    local prefix=$3
    local cnf_or_bb=$4  # "cnf" o "backbone"
    
    # Determinar prefijo de directorio dentro del tar
    # Para "backbone" usar "bb", para "cnf" usar "cnf"
    local dir_prefix="$cnf_or_bb"
    if [ "$cnf_or_bb" == "backbone" ]; then
        dir_prefix="bb"
    fi
    
    # Intenta en archivo suelto primero
    if [ -f "$DATA_DIR/${cnf_or_bb}/$SET_TYPE/$filename" ]; then
        cp "$DATA_DIR/${cnf_or_bb}/$SET_TYPE/$filename" "$dest_dir/"
        return 0
    fi
    
    # Intenta en TAR.GZ "orig" (sin prefijo dual)
    local tar_file="$DATA_DIR/${cnf_or_bb}/$SET_TYPE/${dir_prefix}_${prefix}.tar.gz"
    if [ -f "$tar_file" ]; then
        # Intento 1: Buscar el archivo directamente en la raíz del TAR
        if tar -tzf "$tar_file" "$filename" &>/dev/null; then
            tar -xzf "$tar_file" -C "$dest_dir/" "$filename" 2>/dev/null
            if [ $? -eq 0 ]; then
                return 0
            fi
        fi
        
        # Intento 2: Buscar dentro de la carpeta intermedia SIN ./
        # (ej: cnf_pt/filename o bb_pt/filename)
        local path_without_dot="${dir_prefix}_${prefix}/$filename"
        if tar -tzf "$tar_file" "$path_without_dot" &>/dev/null; then
            tar -xzf "$tar_file" -C "$dest_dir/" "$path_without_dot"
            local extract_status=$?
            if [ -f "$dest_dir/$path_without_dot" ]; then
                mv "$dest_dir/$path_without_dot" "$dest_dir/" 2>/dev/null
                if [ -d "$dest_dir/${dir_prefix}_${prefix}" ]; then
                    rmdir "$dest_dir/${dir_prefix}_${prefix}" 2>/dev/null
                fi
                return 0
            fi
        fi
        
        # Intento 3: Buscar dentro de la carpeta intermedia CON ./
        # (ej: ./cnf_pt/filename o ./bb_pt/filename)
        local path_with_dot="./${dir_prefix}_${prefix}/$filename"
        if tar -tzf "$tar_file" "$path_with_dot" &>/dev/null; then
            tar -xzf "$tar_file" -C "$dest_dir/" "$path_with_dot"
            local extract_status=$?
            # tar extrae con los ./ en la ruta, así que buscar así
            if [ -f "$dest_dir/.${dir_prefix}_${prefix}/$filename" ]; then
                mv "$dest_dir/.${dir_prefix}_${prefix}/$filename" "$dest_dir/" 2>/dev/null
                if [ -d "$dest_dir/.${dir_prefix}_${prefix}" ]; then
                    rmdir "$dest_dir/.${dir_prefix}_${prefix}" 2>/dev/null
                fi
                return 0
            elif [ -f "$dest_dir/${dir_prefix}_${prefix}/$filename" ]; then
                mv "$dest_dir/${dir_prefix}_${prefix}/$filename" "$dest_dir/" 2>/dev/null
                if [ -d "$dest_dir/${dir_prefix}_${prefix}" ]; then
                    rmdir "$dest_dir/${dir_prefix}_${prefix}" 2>/dev/null
                fi
                return 0
            fi
        fi
    fi
    
    # Intenta en TAR.GZ "dual"
    local tar_file_dual="$DATA_DIR/${cnf_or_bb}/$SET_TYPE/d_${dir_prefix}_${prefix}.tar.gz"
    if [ -f "$tar_file_dual" ]; then
        # Intento 1: Buscar el archivo directamente en la raíz del TAR
        if tar -tzf "$tar_file_dual" "$filename" &>/dev/null; then
            tar -xzf "$tar_file_dual" -C "$dest_dir/" "$filename" 2>/dev/null
            if [ $? -eq 0 ]; then
                return 0
            fi
        fi
        
        # Intento 1b: Buscar con prefijo d_ en el archivo
        local filename_with_d="d_$filename"
        if tar -tzf "$tar_file_dual" "$filename_with_d" &>/dev/null; then
            tar -xzf "$tar_file_dual" -C "$dest_dir/" "$filename_with_d"
            if [ -f "$dest_dir/$filename_with_d" ]; then
                mv "$dest_dir/$filename_with_d" "$dest_dir/$filename" 2>/dev/null
                return 0
            fi
        fi
        
        # Intento 2: Buscar dentro de la carpeta intermedia SIN ./
        # (ej: d_cnf_pt/filename o d_bb_pt/filename)
        local path_without_dot="d_${dir_prefix}_${prefix}/$filename"
        if tar -tzf "$tar_file_dual" "$path_without_dot" &>/dev/null; then
            tar -xzf "$tar_file_dual" -C "$dest_dir/" "$path_without_dot"
            local extract_status=$?
            if [ -f "$dest_dir/$path_without_dot" ]; then
                mv "$dest_dir/$path_without_dot" "$dest_dir/" 2>/dev/null
                if [ -d "$dest_dir/d_${dir_prefix}_${prefix}" ]; then
                    rmdir "$dest_dir/d_${dir_prefix}_${prefix}" 2>/dev/null
                fi
                return 0
            fi
        fi
        
        # Intento 2b: Buscar dentro de la carpeta con prefijo d_ en el archivo
        local path_without_dot_d="d_${dir_prefix}_${prefix}/$filename_with_d"
        if tar -tzf "$tar_file_dual" "$path_without_dot_d" &>/dev/null; then
            tar -xzf "$tar_file_dual" -C "$dest_dir/" "$path_without_dot_d"
            if [ -f "$dest_dir/$path_without_dot_d" ]; then
                mv "$dest_dir/$path_without_dot_d" "$dest_dir/$filename" 2>/dev/null
                if [ -d "$dest_dir/d_${dir_prefix}_${prefix}" ]; then
                    rmdir "$dest_dir/d_${dir_prefix}_${prefix}" 2>/dev/null
                fi
                return 0
            fi
        fi
        
        # Intento 3: Buscar dentro de la carpeta intermedia CON ./
        # (ej: ./d_cnf_pt/filename o ./d_bb_pt/filename)
        local path_with_dot="./d_${dir_prefix}_${prefix}/$filename"
        if tar -tzf "$tar_file_dual" "$path_with_dot" &>/dev/null; then
            tar -xzf "$tar_file_dual" -C "$dest_dir/" "$path_with_dot"
            local extract_status=$?
            if [ -f "$dest_dir/.d_${dir_prefix}_${prefix}/$filename" ]; then
                mv "$dest_dir/.d_${dir_prefix}_${prefix}/$filename" "$dest_dir/" 2>/dev/null
                if [ -d "$dest_dir/.d_${dir_prefix}_${prefix}" ]; then
                    rmdir "$dest_dir/.d_${dir_prefix}_${prefix}" 2>/dev/null
                fi
                return 0
            elif [ -f "$dest_dir/d_${dir_prefix}_${prefix}/$filename" ]; then
                mv "$dest_dir/d_${dir_prefix}_${prefix}/$filename" "$dest_dir/" 2>/dev/null
                if [ -d "$dest_dir/d_${dir_prefix}_${prefix}" ]; then
                    rmdir "$dest_dir/d_${dir_prefix}_${prefix}" 2>/dev/null
                fi
                return 0
            fi
        fi
        
        # Intento 3b: Con prefijo d_ en el archivo CON ./
        local path_with_dot_d="./d_${dir_prefix}_${prefix}/$filename_with_d"
        if tar -tzf "$tar_file_dual" "$path_with_dot_d" &>/dev/null; then
            tar -xzf "$tar_file_dual" -C "$dest_dir/" "$path_with_dot_d"
            if [ -f "$dest_dir/.d_${dir_prefix}_${prefix}/$filename_with_d" ]; then
                mv "$dest_dir/.d_${dir_prefix}_${prefix}/$filename_with_d" "$dest_dir/$filename" 2>/dev/null
                if [ -d "$dest_dir/.d_${dir_prefix}_${prefix}" ]; then
                    rmdir "$dest_dir/.d_${dir_prefix}_${prefix}" 2>/dev/null
                fi
                return 0
            elif [ -f "$dest_dir/d_${dir_prefix}_${prefix}/$filename_with_d" ]; then
                mv "$dest_dir/d_${dir_prefix}_${prefix}/$filename_with_d" "$dest_dir/$filename" 2>/dev/null
                if [ -d "$dest_dir/d_${dir_prefix}_${prefix}" ]; then
                    rmdir "$dest_dir/d_${dir_prefix}_${prefix}" 2>/dev/null
                fi
                return 0
            fi
        fi
    fi
    
    return 1  # No encontrado
}

# ==============================================================================
# FUNCIÓN: Buscar archivo backbone con fallbacks
# Args: $1=cnf_filename, $2=destino, $3=prefix
# ==============================================================================
find_and_copy_backbone() {
    local cnf_name=$1
    local dest_dir=$2
    local prefix=$3
    
    # Variante 1: cnf_name.backbone.xz (ej: archivo.cnf.xz.backbone.xz)
    # Agregar .backbone.xz al nombre completo del CNF
    local bb_name="${cnf_name}.backbone.xz"
    if extract_from_tar "$bb_name" "$dest_dir" "$prefix" "backbone"; then
        return 0
    fi
    
    # Variante 2: nombre del CNF sin la última extensión
    # Ej: vlsat_49200_7490695.mcc2020_cnf.bz2 → vlsat_49200_7490695.mcc2020_cnf.backbone.xz
    local base_name=$(echo "$cnf_name" | sed 's/\.[a-z0-9]*$//')
    bb_name="${base_name}.backbone.xz"
    if extract_from_tar "$bb_name" "$dest_dir" "$prefix" "backbone"; then
        return 0
    fi
    
    echo "  [Advertencia] No se encontró backbone para: $cnf_name"
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

echo "Procesando archivos del batch (PREFIX=$PREFIX)..."

# Leer archivo batch y procesar cada línea
processed_count=0
failed_count=0
while IFS= read -r cnf_filename; do
    # Trimear espacios y caracteres especiales (ej: ~)
    cnf_filename=$(echo "$cnf_filename" | sed 's/^[[:space:]~]*//' | sed 's/[[:space:]]*$//')
    
    # Ignorar líneas vacías
    [ -z "$cnf_filename" ] && continue
    
    echo "  ✓ Extrayendo: $cnf_filename"
    
    # Extraer CNF
    if extract_from_tar "$cnf_filename" "$LOCAL_TMP/input" "$PREFIX" "cnf"; then
        ((processed_count++))
    else
        echo "    [ERROR] No se encontró CNF: $cnf_filename"
        ((failed_count++))
        continue
    fi
    
    # Extraer backbone (no falla si no existe)
    find_and_copy_backbone "$cnf_filename" "$LOCAL_TMP/backbone" "$PREFIX"
    
done < "$BATCH_FILE"

echo "Resumen: $processed_count archivos procesados, $failed_count errores"

# Crear archivo limpio para python
cat "$BATCH_FILE" | grep -v '^$' > "$LOCAL_TMP/clean_batch.txt"

# Pre-creamos subcarpetas para evitar errores de multiprocessing en python
mkdir -p "$LOCAL_TMP/output/processed"

# ==============================================================================
# EJECUCIÓN DE graph.py
# ==============================================================================
echo "Ejecutando graph.py..."

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
# COPIAR RESULTADOS
# ==============================================================================
echo "Guardando resultados..."

# Ubicación destino depende del entorno
if [ "$RUNNING_ENV" == "savio" ]; then
    RESULTS_DIR="$DATA_DIR/pt/$SET_TYPE/processed"
else
    # En local, guardar en data/
    RESULTS_DIR="$DATA_DIR/processed/$SET_TYPE"
fi

mkdir -p "$RESULTS_DIR"

if [ -d "$LOCAL_TMP/output/processed" ]; then
    rsync -a "$LOCAL_TMP/output/processed/" "$RESULTS_DIR/"
    echo "Resultados guardados en: $RESULTS_DIR"
else
    echo "Advertencia: No hay resultados en $LOCAL_TMP/output/processed"
fi

# ==============================================================================
# LIMPIEZA
# ==============================================================================
echo "Limpiando directorios temporales..."
rm -rf "$LOCAL_TMP"

echo "#=== Tarea completada ===#"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch completo"
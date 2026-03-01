#!/bin/bash
# Crear carpetas para las listas
mkdir -p batches/pretrain batches/finetune batches/validation

# Función para generar batches
generate_batches() {
    local folder=$1
    local size=$2
    ls /global/home/users/$USER/scratch/neuroback/data/cnf/$folder/*.xz | xargs -n 1 basename > all_${folder}.txt
    split -l $size all_${folder}.txt batches/${folder}/batch_ --additional-suffix=.txt -d
}

# Generamos: batches de 5000 para pretrain, 500 para finetune y 500 para validation
generate_batches "pretrain" 5000
generate_batches "finetune" 500
generate_batches "validation" 500
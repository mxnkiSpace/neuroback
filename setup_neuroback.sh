#!/bin/bash

# Definimos rutas
BASE_DIR="/global/home/users/$USER/scratch/neuroback"
DATA_DIR="$BASE_DIR/data"
REPO_DIR=$(pwd) 

echo "Creando estructura en Scratch: $BASE_DIR..."
mkdir -p $DATA_DIR/cnf/{pretrain,finetune,validation,test}
mkdir -p $DATA_DIR/backbone/{pretrain,finetune,validation}

# Movemos los datos del repo a scratch
echo "Copiando datos locales del repo a Scratch..."
cp -r $REPO_DIR/data/cnf/validation/* $DATA_DIR/cnf/validation/ 2>/dev/null
cp -r $REPO_DIR/data/cnf/test/* $DATA_DIR/cnf/test/ 2>/dev/null
cp -r $REPO_DIR/data/backbone/validation/* $DATA_DIR/backbone/validation/ 2>/dev/null

# Descargamos datos
echo "Descargando datoscde HuggingFace..."
cd $BASE_DIR

wget -O cnf_pt_orig.tar.gz https://huggingface.co/datasets/neuroback/DataBack/resolve/main/original/cnf_pt.tar.gz
wget -O bb_pt_orig.tar.gz https://huggingface.co/datasets/neuroback/DataBack/resolve/main/original/bb_pt.tar.gz
wget -O cnf_ft_orig.tar.gz https://huggingface.co/datasets/neuroback/DataBack/resolve/main/original/cnf_ft.tar.gz
wget -O bb_ft_orig.tar.gz https://huggingface.co/datasets/neuroback/DataBack/resolve/main/original/bb_ft.tar.gz

wget -O cnf_pt_dual.tar.gz https://huggingface.co/datasets/neuroback/DataBack/resolve/main/dual/cnf_pt.tar.gz
wget -O bb_pt_dual.tar.gz https://huggingface.co/datasets/neuroback/DataBack/resolve/main/dual/bb_pt.tar.gz
wget -O cnf_ft_dual.tar.gz https://huggingface.co/datasets/neuroback/DataBack/resolve/main/dual/cnf_ft.tar.gz
wget -O bb_ft_dual.tar.gz https://huggingface.co/datasets/neuroback/DataBack/resolve/main/dual/bb_ft.tar.gz

# Descomprimimos los datos descargados
echo "Datos descargados correctamente, ahora los descomprimimos..."
tar -xzvf cnf_pt_orig.tar.gz -C $DATA_DIR/cnf/pretrain/
tar -xzvf bb_pt_orig.tar.gz -C $DATA_DIR/backbone/pretrain/
tar -xzvf cnf_ft_orig.tar.gz -C $DATA_DIR/cnf/finetune/
tar -xzvf bb_ft_orig.tar.gz -C $DATA_DIR/backbone/finetune/

tar -xzvf cnf_pt_dual.tar.gz -C $DATA_DIR/cnf/pretrain/
tar -xzvf bb_pt_dual.tar.gz -C $DATA_DIR/backbone/pretrain/
tar -xzvf cnf_ft_dual.tar.gz -C $DATA_DIR/cnf/finetune/
tar -xzvf bb_ft_dual.tar.gz -C $DATA_DIR/backbone/finetune/

# Limpiamos los archivos comprimidos
echo "Finalizando configuracion..."
rm *.tar.gz

# Creamos un enlace para que el código vea la carpeta 'data' de scratch como si fuera local
ln -sfn $DATA_DIR $REPO_DIR/data

echo "TODO LISTO. Los datos están en $DATA_DIR y vinculados a $REPO_DIR/data_scratch"
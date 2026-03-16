#!/bin/bash

wget -O cnf_pt.tar.gz https://huggingface.co/datasets/neuroback/DataBack/resolve/main/original/cnf_pt.tar.gz && echo "¡Archivo 1 descargado correctamente!"
wget -O bb_pt.tar.gz https://huggingface.co/datasets/neuroback/DataBack/resolve/main/original/bb_pt.tar.gz && echo "¡Archivo 2 descargado correctamente!"
wget -O cnf_ft.tar.gz https://huggingface.co/datasets/neuroback/DataBack/resolve/main/original/cnf_ft.tar.gz && echo "¡Archivo 3 descargado correctamente!"
wget -O bb_ft.tar.gz https://huggingface.co/datasets/neuroback/DataBack/resolve/main/original/bb_ft.tar.gz && echo "¡Archivo 4 descargado correctamente!"

wget -O d_cnf_pt.tar.gz https://huggingface.co/datasets/neuroback/DataBack/resolve/main/dual/d_cnf_pt.tar.gz && echo "¡Archivo 5 descargado correctamente!"
wget -O d_bb_pt.tar.gz https://huggingface.co/datasets/neuroback/DataBack/resolve/main/dual/d_bb_pt.tar.gz && echo "¡Archivo 6 descargado correctamente!"
wget -O d_cnf_ft.tar.gz https://huggingface.co/datasets/neuroback/DataBack/resolve/main/dual/d_cnf_ft.tar.gz && echo "¡Archivo 7 descargado correctamente!"
wget -O d_bb_ft.tar.gz https://huggingface.co/datasets/neuroback/DataBack/resolve/main/dual/d_bb_ft.tar.gz && echo "¡Archivo 8 descargado correctamente!"


mv cnf_pt.tar.gz data/cnf/pretrain
mv bb_pt.tar.gz data/backbone/pretrain
mv cnf_ft.tar.gz data/cnf/finetune
mv bb_ft.tar.gz data/backbone/finetune

mv d_cnf_pt.tar.gz data/cnf/pretrain
mv d_bb_pt.tar.gz data/backbone/pretrain
mv d_cnf_ft.tar.gz data/cnf/finetune
mv d_bb_ft.tar.gz data/backbone/finetune
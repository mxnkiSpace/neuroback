# Preparación para correr en Clúster
## Clonar y crear Entorno

```bash
git clone https://github.com/mxnkiSpace/neuroback.git
cd neuroback

# Crear y activar entorno
python3 -m venv /global/home/users/$USER/scratch/neuroback/.venv

source /global/home/users/$USER/scratch/neuroback/.venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Preparar datos
Debe correrse en un nodo con acceso a internet
```bash
chmod +x setup_neuroback.sh
./setup_neuroback.sh
```

## Generación de Gráfos
```bash
sbatch job_generate_graphs.sh
```
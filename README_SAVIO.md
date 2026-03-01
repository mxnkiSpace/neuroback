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
## Generacion de Batches
```bash
chmod +x batch_generation.sh
./batch_generation.sh
```
## Prueba

```bash
sbatch --array=0 job_generate_graphs.sh pretrain
```

## Generación de Gráfos
### Grafos pretrain
```bash
sbatch --array=1-23 job_generate_graphs.sh pretrain
```

### Grafos finetuning
```bash
sbatch --array=0-3 job_generate_graphs.sh finetune
```
### Grafos validation
```bash
sbatch --array=0-0 job_generate_graphs.sh validation
```

# Ejecución en SAVIO Cluster

Este documento describe cómo ejecutar Neuroback en el cluster SAVIO. El código soporta tanto ejecución local (para pruebas) como en el cluster.

---

## 1. Preparación inicial

### Clonar repositorio y crear entorno virtual

```bash
# En el nodo login o en tu directorio local
git clone https://github.com/mxnkiSpace/neuroback.git
cd neuroback

# Crear entorno virtual en Scratch (espacio permanente en SAVIO)
python3 -m venv /global/home/users/$USER/scratch/neuroback/.venv

# Activar entorno
source /global/home/users/$USER/scratch/neuroback/.venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Preparar datos

Ejecutar este comando en un nodo con acceso a internet:

```bash
chmod +x setup_neuroback.sh
./setup_neuroback.sh
```

Esto descargará y organizará los datos en:
```
data/
├── cnf/
│   ├── pretrain/   (24 batches)
│   ├── finetune/   (4 batches)
│   └── validation/ (1 batch)
└── backbone/
    ├── pretrain/
    ├── finetune/
    └── validation/
```

**Nota**: Los archivos se descomprimirán dinámicamente cuando se procesen, no es necesario descomprimirlos globalmente.

---

## 2. Ejecución en SAVIO

El script [job_generate_graphs.sh](job_generate_graphs.sh) **detecta automáticamente** que está en SAVIO y ajusta las rutas. No es necesario modificar nada.

### Prueba rápida

Ejecutar un único batch para verificar que todo funciona:

```bash
sbatch --array=0 job_generate_graphs.sh validation
```

Verificar el estado:
```bash
squeue -u $USER
```

Revisar los logs:
```bash
ls -lh logs/nb_*.out
cat logs/nb_JOBID_0.out  # Reemplaza JOBID con el ID del trabajo
```

### Generación completa de gráfos

#### Gráfos de preentrenamiento (24 batches)
```bash
sbatch --array=0-23 job_generate_graphs.sh pretrain
```

#### Gráfos de fine-tuning (4 batches)
```bash
sbatch --array=0-3 job_generate_graphs.sh finetune
```

#### Gráfos de validación (1 batch)
```bash
sbatch --array=0 job_generate_graphs.sh validation
```

#### Todos en paralelo (28 trabajos simultáneamente)
```bash
sbatch --array=0-23 job_generate_graphs.sh pretrain
sbatch --array=0-3 job_generate_graphs.sh finetune
sbatch --array=0 job_generate_graphs.sh validation
```

---

## 3. Ejecución Local (para pruebas)

También se puede ejecutar el código en local para pruebas. El script detectará automáticamente que NO está en SAVIO y ajustará las rutas.

### Requisitos

```bash
# Clonar repositorio
cd ~/path/a/neuroback

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Prueba un batch

```bash
bash job_generate_graphs.sh validation 0
bash job_generate_graphs.sh pretrain 5
```

Resultados en:
```
data/processed/validation/
data/processed/pretrain/
```

---

## 4. Monitoreo y troubleshooting

### Ver estado de trabajos

```bash
# Todos tus trabajos
squeue -u $USER

# Detalles de un trabajo específico
scontrol show job JOBID
```

### Cancelar trabajos

```bash
# Un trabajo
scancel JOBID

# Todos en un array
scancel JOBID_*
```

### Ver logs

```bash
# Los logs se guardan en logs/nb_JOBID_TASKID.out
ls -lh logs/
cat logs/nb_12345678_0.out
```

## 5 Estructura de salida

### En SAVIO

```
$SCRATCH/neuroback/neuroback/data/pt/{tipo}/processed/
├── archivo.cnf.xz.c-0.pt
├── archivo.cnf.xz.c-1.pt
└── ...
```

### En local

```
./data/processed/{tipo}/
├── archivo.cnf.xz.c-0.pt
├── archivo.cnf.xz.c-1.pt
└── ...
```

Cada archivo `.pt` contiene un grafo PyTorch serializado (formato de Graph Neural Network).

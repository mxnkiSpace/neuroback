import os
import tarfile
import urllib.request

# 1. Configuración de Batches
BATCH_SIZES = {
    "pretrain": 5000,
    "finetune": 500
}

# 2. URLs corregidas según la estructura exacta de HuggingFace
URLS = {
    "pretrain": {
        "orig": "https://huggingface.co/datasets/neuroback/DataBack/resolve/main/original/cnf_pt.tar.gz",
        "dual": "https://huggingface.co/datasets/neuroback/DataBack/resolve/main/dual/d_cnf_pt.tar.gz"
    },
    "finetune": {
        "orig": "https://huggingface.co/datasets/neuroback/DataBack/resolve/main/original/cnf_ft.tar.gz",
        "dual": "https://huggingface.co/datasets/neuroback/DataBack/resolve/main/dual/d_cnf_ft.tar.gz"
    }
}

def download_file(url, filename):
    """Descarga el archivo solo si no existe localmente."""
    if not os.path.exists(filename):
        print(f"Descargando {filename} (esto puede tomar un tiempo)...")
        try:
            urllib.request.urlretrieve(url, filename)
        except urllib.error.HTTPError as e:
            print(f"Error HTTP {e.code}: No se pudo descargar desde {url}")
            raise e
    else:
        print(f"El archivo {filename} ya existe. Omitiendo descarga.")

def analyze_and_batch():
    # Crear carpetas de salida
    os.makedirs("batches/pretrain", exist_ok=True)
    os.makedirs("batches/finetune", exist_ok=True)
    os.makedirs("batches/validation", exist_ok=True)

    for dataset_type, sources in URLS.items():
        print(f"\n--- Procesando {dataset_type.upper()} ---")
        all_files = []
        
        for source_name, url in sources.items(): # source_name será 'orig' o 'dual'
            # Usamos el nombre exacto que viene en la URL para guardarlo localmente
            tar_filename = url.split('/')[-1]
            download_file(url, tar_filename)
            
            print(f"Leyendo metadatos de {tar_filename}...")
            # Leemos el índice del tar.gz para extraer nombres y tamaños sin descomprimir
            with tarfile.open(tar_filename, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.isfile() and member.name.endswith(".xz"):
                        all_files.append({
                            'source': source_name,
                            'name': member.name,
                            'size': member.size
                        })
        
        # 2. Ordenar estrictamente por tamaño (de mayor a menor)
        all_files.sort(key=lambda x: x['size'], reverse=True)
        print(f"Total de archivos detectados en {dataset_type}: {len(all_files)}")
        
        # 3. Generar los archivos .txt de batch
        batch_size = BATCH_SIZES[dataset_type]
        for i in range(0, len(all_files), batch_size):
            batch_slice = all_files[i : i + batch_size]
            batch_idx = i // batch_size
            batch_filename = f"batches/{dataset_type}/batch_{batch_idx:02d}.txt"
            
            with open(batch_filename, "w") as f:
                for item in batch_slice:
                    # Formato: [orig/dual] [nombre_del_archivo]
                    f.write(f"{item['source']}\t{item['name']}\n")
        
        print(f"Se generaron {batch_idx + 1} batches para {dataset_type}.")

    # 4. Manejo de Validation (archivos sueltos)
    val_dir = "data/cnf/validation"
    if os.path.exists(val_dir):
        print("\n--- Procesando VALIDATION ---")
        val_files = [f for f in os.listdir(val_dir) if f.endswith(".xz")]
        if val_files:
            with open("batches/validation/batch_00.txt", "w") as f:
                for val_f in val_files:
                    f.write(f"local\t{val_f}\n")
            print(f"Se generó 1 batch para validation con {len(val_files)} archivos.")
        else:
            print("No se encontraron archivos .xz en la carpeta validation local.")

if __name__ == "__main__":
    analyze_and_batch()
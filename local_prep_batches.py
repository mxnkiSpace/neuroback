import os
import tarfile
import urllib.request

# 1. Configuración de Batches
BATCH_SIZES = {
    "pretrain": 5000,
    "finetune": 500
}

PATHS = {
    "pretrain": "data/cnf/pretrain",
    "finetune": "data/cnf/finetune"
}

# Paths donde se encuentran los datos. Los archivos ya están descargados
URLS = {
    "pretrain": {
        "orig": "data/cnf/pretrain/cnf_pt.tar.gz",
        "dual": "data/cnf/pretrain/d_cnf_pt.tar.gz"
    },
    "finetune": {
        "orig": "data/cnf/finetune/cnf_ft.tar.gz",
        "dual": "data/cnf/finetune/d_cnf_ft.tar.gz"
    }
}

def generate_batches_pretrain():
    names = []

    #Añademos los arhivos sueltos que estan en 
    for root, _, files in os.walk(PATHS["pretrain"]):
        for file in files:
            if file.endswith(".xz") or file.endswith(".cnf-1.gz"):
                file_path = os.path.join(root, file)
                names.append({
                    'name': file,
                    'size': os.path.getsize(file_path)
                })

    # Los nombres quedan de la forma ./d_cnf_pt/d_CliqueFormula_56_10_35.cnf.xz y ./cnf_pt/CliqueFormula_62_10_39.cnf.xz
    os.makedirs("batches/pretrain", exist_ok=True)
    urls = URLS["pretrain"]
    for _, url in urls.items():
        with tarfile.open(url, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    names.append({
                        # Le quitamos el prefijo "./d_cnf_pt/" o "./cnf_pt/" para que quede solo el nombre del archivo
                        'name': member.name.split("/")[-1],
                        'size': member.size
                    })
    # Ordenamos los archivos
    names.sort(key=lambda x: x['size'], reverse=True)

    #Generamos los batches
    batch_size = BATCH_SIZES["pretrain"]
    for i in range(0, len(names), batch_size):
        batch_slice = names[i : i + batch_size]
        batch_idx = i // batch_size
        os.makedirs("batches/pretrain", exist_ok=True)
        with open(f"batches/pretrain/batch_{batch_idx:02d}.txt", "w") as f:
            for item in batch_slice:
                f.write(f"{item['name']}\n")   
    

def generate_batches_finetune():
    names = []

    #Añademos los arhivos sueltos que estan en 
    for root, _, files in os.walk(PATHS["finetune"]):
        for file in files:
            file_path = os.path.join(root, file)
            names.append({
                'name': file,
                'size': os.path.getsize(file_path)
            })

    os.makedirs("batches/finetune", exist_ok=True)
    urls = URLS["finetune"]
    for _, url in urls.items():
        with tarfile.open(url, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    names.append({
                        'name': member.name.split("/")[-1],
                        'size': member.size
                    })
    # Ordenamos los archivos
    names.sort(key=lambda x: x['size'], reverse=True)

    #Generamos los batches
    batch_size = BATCH_SIZES["finetune"]
    for i in range(0, len(names), batch_size):
        batch_slice = names[i : i + batch_size]
        batch_idx = i // batch_size
        os.makedirs("batches/finetune", exist_ok=True)
        with open(f"batches/finetune/batch_{batch_idx:02d}.txt", "w") as f:
            for item in batch_slice:
                f.write(f"{item['name']}\n")

def generate_batches_validation():
    # Para validación, simplemente listamos los archivos sin batch
    names = []
    for root, _, files in os.walk("data/cnf/validation"):
        for file in files:
            file_path = os.path.join(root, file)
            names.append(file)

    os.makedirs("batches/validation", exist_ok=True)
    with open("batches/validation/batch_00.txt", "w") as f:
        for name in names:
            f.write(f"{name}\n")


if __name__ == "__main__":
    generate_batches_pretrain()
    print("Batches de pretrain generados.")
    generate_batches_finetune()
    print("Batches de finetune generados.")
    generate_batches_validation()
    print("Batches de validación generados.")
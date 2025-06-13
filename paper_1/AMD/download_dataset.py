import kagglehub
import os
import shutil

download_dir = "/home/data3/Ali/Code/Saina/Brea/Dataset/"

os.makedirs(download_dir, exist_ok=True)

# Descarga el dataset
path = kagglehub.dataset_download("paultimothymooney/breast-histopathology-images")

print(f"Dataset descargado en: {path}")

# Mover contenido del dataset a la carpeta destino
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(download_dir, item)
    if os.path.isdir(s):
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)

print(f"Contenido movido a: {download_dir}")
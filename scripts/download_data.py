from datasets import load_dataset
import os

print("Descargando datos desde Hugging Face...")
dataset = load_dataset(
    "json",
    data_files="hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/Video_Games.jsonl",
    split="train"
)

# Reducir el dataset
print("Reduciendo el dataset al 25%...")
# Usamos select y range para tomar la mitad de las filas de forma rápida
# Si quieres algo aleatorio usa: dataset = dataset.shuffle(seed=42).select(range(len(dataset)//2))
dataset = dataset.select(range(len(dataset) // 4)) 
# --------------------------------------

# Asegurarnos de que la carpeta de destino de Kedro existe
os.makedirs("data/01_raw", exist_ok=True)

print(f"Guardando {len(dataset)} filas en data/01_raw/video_games.parquet...")
dataset.to_parquet("data/01_raw/video_games.parquet")

print("¡Descarga y guardado reducidos completados con éxito!")
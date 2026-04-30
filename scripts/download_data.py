from datasets import load_dataset
import os

print("Descargando datos directamente desde el archivo JSON original...")
dataset = load_dataset(
    "json",
    data_files="hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/Video_Games.jsonl",
    split="train"
)

# Asegurarnos de que la carpeta de destino de Kedro existe
os.makedirs("data/01_raw", exist_ok=True)

print("Guardando en data/01_raw/video_games.parquet...")
# Al ejecutar el script desde la raíz del proyecto, la ruta correcta es esta:
dataset.to_parquet("data/01_raw/video_games.parquet")

print("¡Descarga y guardado completados con éxito!")
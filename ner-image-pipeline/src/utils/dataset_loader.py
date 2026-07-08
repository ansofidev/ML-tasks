import os
import zipfile
import subprocess
import json
from datasets import Dataset


def download_animals10(dataset_path="data/cv/raw-img"):
    # Skip download if dataset already exists
    if os.path.exists(dataset_path):
        print("Dataset already exists.")
        return

    print("Downloading Animals10 dataset...")

    # Download dataset from Kaggle
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "alessiocorrado99/animals10"],
        check=True
    )

    # Create target directory
    os.makedirs("data/cv", exist_ok=True)

    # Extract downloaded zip archive
    with zipfile.ZipFile("animals10.zip", "r") as zip_ref:
        zip_ref.extractall("data/cv")

    print("Dataset ready.")


# # Load dataset and transform it to HuggingFace Dataset format
def load_dataset_for_ner(path):
    with open(path) as f:
        data = json.load(f)

    return Dataset.from_list(data)
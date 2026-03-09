import kagglehub
import shutil
import os
from pathlib import Path
from collections import defaultdict

def download_dataset():
    path = kagglehub.dataset_download("sreeharims/glaucoma-dataset")
    dest = Path("datasets/LAG")
    if not dest.exists():
        shutil.copytree(path, dest)
        print(f"Dataset copiato in: {dest}")
    else:
        print(f"Dataset già presente in: {dest}")
    return dest

def explore(root: Path):
    print(f"\n{'='*50}")
    print(f"Root: {root}")
    print(f"{'='*50}")

    for dirpath, dirnames, filenames in os.walk(root):
        depth = str(dirpath).replace(str(root), "").count(os.sep)
        if depth > 3:
            continue
        indent = "  " * depth
        print(f"{indent}{Path(dirpath).name}/")
        if filenames:
            sub = "  " * (depth + 1)
            for f in filenames[:5]:
                print(f"{sub}{f}")
            if len(filenames) > 5:
                print(f"{sub}... ({len(filenames)} file totali)")

    print(f"\n{'='*50}")
    print("Conteggio per cartella:")
    print(f"{'='*50}")
    class_counts = defaultdict(int)
    extensions = defaultdict(int)

    for dirpath, _, filenames in os.walk(root):
        images = [f for f in filenames if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]
        if images:
            rel = Path(dirpath).relative_to(root)
            class_counts[str(rel)] = len(images)
            for f in images:
                extensions[Path(f).suffix.lower()] += 1

    for folder, count in sorted(class_counts.items()):
        print(f"  {folder:<50} {count:>6} immagini")

    print(f"\nEstensioni trovate: {dict(extensions)}")
    print(f"Totale immagini:    {sum(class_counts.values())}")
import pandas as pd

def explore_labels(root: Path):
    print(f"\n{'='*50}")
    print("Cercando file di etichette...")
    print(f"{'='*50}")

    # Cerca tutti i file non-immagine
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                full = Path(dirpath) / f
                print(f"  Trovato: {full}")

                # Se è un CSV, stampane le prime righe
                if f.endswith(".csv"):
                    df = pd.read_csv(full)
                    print(f"  Shape: {df.shape}")
                    print(f"  Colonne: {list(df.columns)}")
                    print(df.head())
def explore_filenames(root: Path):
    print(f"\n{'='*50}")
    print("Analisi nomi file:")
    print(f"{'='*50}")

    for split in ["train", "validation", "test"]:
        split_path = root / "LAG" / split
        if not split_path.exists():
            continue

        files = list(split_path.glob("*.jpg"))
        prefixes = defaultdict(int)

        for f in files:
            # Prende il prefisso prima del punto
            # es. "g.0005.jpg" → "g"
            # es. "n.0005.jpg" → "n"
            prefix = f.stem.split(".")[0]
            prefixes[prefix] += 1

        print(f"\n  {split}/")
        for prefix, count in sorted(prefixes.items()):
            print(f"    prefisso '{prefix}' → {count} immagini")

if __name__ == "__main__":
    root = download_dataset()
    explore(root)
    explore_labels(root)
    explore_filenames(root)
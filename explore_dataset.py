import kagglehub
import shutil
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd

def download_dataset():
    path = kagglehub.dataset_download("sreeharims/glaucoma-dataset")
    dest = Path("datasets/")
    if not dest.exists():
        shutil.copytree(path, dest)
        print(f"Dataset copiato in: {dest}")
    else:
        print(f"Dataset già presente in: {dest}")
    return dest

def remove_test_split(root: Path):
    """
    Sposta le immagini dalla cartella 'test' a 'train' e poi elimina 'test'.
    """
    print(f"\n{'='*50}")
    print("Riorganizzazione split: unione di 'test' in 'train'...")
    print(f"{'='*50}")
    
    test_dir = root / "LAG" / "test"
    train_dir = root / "LAG" / "train"
    
    if test_dir.exists():
        # Ci assicuriamo che la cartella train esista
        train_dir.mkdir(parents=True, exist_ok=True)
        
        # Sposta tutti i file
        moved_count = 0
        for item in test_dir.glob("*.jpg"): # Sposta solo le immagini jpg
            if item.is_file():
                shutil.move(str(item), str(train_dir / item.name))
                moved_count += 1
        
        # Rimuove la cartella test ormai vuota (ignora eventuali file nascosti/di sistema)
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"  Spostate {moved_count} immagini da 'test' a 'train'.")
        print("  Cartella 'test' eliminata con successo.")
    else:
        print("  Cartella 'test' non trovata. Potrebbe essere già stata unita.")

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


def explore_labels(root: Path):
    print(f"\n{'='*50}")
    print("Cercando file di etichette...")
    print(f"{'='*50}")

    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                full = Path(dirpath) / f
                print(f"  Trovato: {full}")

                if f.endswith(".csv"):
                    df = pd.read_csv(full)
                    print(f"  Shape: {df.shape}")
                    print(f"  Colonne: {list(df.columns)}")
                    print(df.head())


def explore_filenames(root: Path):
    print(f"\n{'='*50}")
    print("Analisi nomi file:")
    print(f"{'='*50}")

    # Rimosso "test" dalla lista di analisi
    for split in ["train", "validation"]:
        split_path = root / "LAG" / split
        if not split_path.exists():
            continue

        files = list(split_path.glob("*.jpg"))
        prefixes = defaultdict(int)

        for f in files:
            prefix = f.stem.split(".")[0]
            prefixes[prefix] += 1

        print(f"\n  {split}/")
        for prefix, count in sorted(prefixes.items()):
            print(f"    prefisso '{prefix}' → {count} immagini")

if __name__ == "__main__":
    root = download_dataset()
    remove_test_split(root) # <- Esegue lo spostamento dei file
    explore(root)
    explore_labels(root)
    explore_filenames(root)
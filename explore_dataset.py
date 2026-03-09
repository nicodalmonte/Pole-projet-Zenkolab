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

if __name__ == "__main__":
    root = download_dataset()
    explore(root)
import os

DL_DATASET_DICT = {
    "ORIGA_ACRIMA_Fundus" : {
        "type" : "cli",
        "required" : ["kaggle"],
        "commands" : [
            "kaggle datasets download sshikamaru/glaucoma-detection",
            "unzip glaucoma-detection.zip -d .",
            "rm glaucoma-detection.zip",
            "mv ORIGA/ORIGA ORIGA",
            "mv glaucoma.csv ORIGA/glaucoma.csv",
        ]
    },
    "REFUGE2" : {
        "type" : "cli",
        "required" : ["kaggle"],
        "commands" : [
            "kaggle datasets download victorlemosml/refuge2",
            "unzip refuge2.zip -d .",
            "rm refuge2.zip",
        ]
    },
    "LAG" : {
        "type" : "cli",
        "required" : ["kaggle"],
        "commands" : [
            "kaggle datasets download sreeharims/glaucoma-dataset",
            "unzip glaucoma-dataset.zip -d .",
            "rm glaucoma-dataset.zip",
        ]
    },
    "JRAIGS" : {
        "type" : "cli",
        "required" : ["kaggle"],
        "commands" : [
            "kaggle datasets download manit2022/jraigs-dataset",
            "unzip jraigs-dataset.zip -d .",
            "rm jraigs-dataset.zip",
            "mv justRAIGS JRAIGS",
            "mv JRAIGS/0/0 JRAIGS/images",
            "rm -rf JRAIGS/0",
            "mv JRAIGS/1/* JRAIGS/images/",
            "rm -rf JRAIGS/1",
            "mv JRAIGS/2/* JRAIGS/images/",
            "rm -rf JRAIGS/2",
            "mv JRAIGS/3/* JRAIGS/images/",
            "rm -rf JRAIGS/3",
            "mv JRAIGS/4/* JRAIGS/images/",
            "rm -rf JRAIGS/4",
            "mv JRAIGS/5/* JRAIGS/images/",
            "rm -rf JRAIGS/5",
        ]
    },
    "G1020" : {
        "type" : "cli",
        "required" : ["kaggle"],
        "commands" : [
            #"kaggle datasets download kiamahmed/glaucoma-fundus-imaging-g1020-splitted",
            #"unzip glaucoma-fundus-imaging-g1020-splitted.zip -d .",
            #"rm glaucoma-fundus-imaging-g1020-splitted.zip",
            "mv Images_splitted G1020",
        ]
    },
    "RIM-ONE" : {
        "type" : "cli",
        "required" : ["kaggle"],
        "commands" : [
            "kaggle datasets download orvile/rim-one-retinal-dataset-for-assessing-glaucoma",
            "unzip rim-one-retinal-dataset-for-assessing-glaucoma.zip -d RIM-ONE",
            "rm rim-one-retinal-dataset-for-assessing-glaucoma.zip",
        ]
    },
    "AIROGSLight" : {
        "type" : "cli",
        "required" : ["kaggle"],
        "commands" : [
            "kaggle datasets download deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2",
            "unzip glaucoma-dataset-eyepacs-airogs-light-v2.zip -d AIROGSLight",
            "rm glaucoma-dataset-eyepacs-airogs-light-v2.zip",
        ]
    },
    "Harvard" : {
        "type" : "kagglehub",
        "required" : ["kagglehub"],
        "slug" : "ayaanakhter/harvard-glaucoma-dataset",
        "subdir" : "processed_data",
    },
    "MultichannelGlaucoma" : {
        "type" : "kagglehub_multichannel",
        "required" : ["kagglehub"],
        "slug" : "deathtrooper/multichannel-glaucoma-benchmark-dataset",
        # Duplicates already covered by other Dataset classes:
        #   ORIGA, G1020, EyePACS-Glaucoma (≈AIRROGS), REFUGE1-train/val (≈REFUGE2)
    },
}




def install(NAME_LIST = None, reinstall_all = True):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(project_root, "data", "datasets")
    if reinstall_all:
        os.system(f'rm -rf "{data_dir}"')

    os.makedirs(data_dir, exist_ok=True)

    def run_in_data_dir(command):
        return os.system(f'cd "{data_dir}" && {command}')

    def install_kagglehub(name, cfg):
        """Download via kagglehub and copy the relevant subfolder to data_dir/<name>/."""
        import shutil
        try:
            import kagglehub
        except ImportError:
            print(f"[Harvard] kagglehub not installed. Run: pip install kagglehub")
            return
        print(f"[{name}] Downloading via kagglehub: {cfg['slug']}")
        dl_path = kagglehub.dataset_download(cfg["slug"])
        src = os.path.join(dl_path, cfg.get("subdir", ""))
        dst = os.path.join(data_dir, name)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"[{name}] Installed to {dst}")

    keys = NAME_LIST if NAME_LIST is not None else list(DL_DATASET_DICT.keys())
    for name in keys:
        if name not in DL_DATASET_DICT:
            print(f"Dataset {name} not found in the dataset dictionary.")
            continue
        cfg = DL_DATASET_DICT[name]
        if cfg.get("type") == "cli":
            for command in cfg["commands"]:
                run_in_data_dir(command)
        elif cfg.get("type") == "kagglehub":
            install_kagglehub(name, cfg)
        elif cfg.get("type") == "kagglehub_multichannel":
            import shutil
            try:
                import kagglehub
            except ImportError:
                print(f"[{name}] kagglehub not installed. Run: pip install kagglehub")
                continue
            print(f"[{name}] Downloading via kagglehub: {cfg['slug']}")
            dl_path = kagglehub.dataset_download(cfg["slug"])
            dst = os.path.join(data_dir, name)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            os.makedirs(dst)
            shutil.copytree(os.path.join(dl_path, "full-fundus", "full-fundus"), os.path.join(dst, "images"))
            shutil.copy(os.path.join(dl_path, "metadata - standardized.csv"), os.path.join(dst, "metadata.csv"))
            print(f"[{name}] Installed to {dst}")

if __name__ == '__main__':
    install(["G1020"], reinstall_all = False)
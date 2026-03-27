import os

DL_DATASET_DICT = {
    "ORIGA_Fudus_ACRIMA" : {
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
}

def install(NAME_LIST = None, reinstall_all = True):
    # Calcola la radice del progetto correttamente (dove si trova questo script)
    project_root = os.path.abspath(os.path.dirname(__file__))
    
    if NAME_LIST is None:
        NAME_LIST = DL_DATASET_DICT.keys()

    for name in NAME_LIST:
        if name in DL_DATASET_DICT:
            if DL_DATASET_DICT[name].get("type", None) == "cli":
                
                # Punta direttamente alla tua cartella esistente "datasets/NOME_DATASET"
                data_dir = os.path.join(project_root, "datasets", name)
                
                # Se devi reinstallare, svuota il contenuto della cartella senza eliminare la cartella stessa
                if reinstall_all:
                    os.system(f'rm -rf "{data_dir}"/*')

                # Esegue i comandi spostandosi nella cartella che hai già creato
                for command in DL_DATASET_DICT[name]["commands"]:
                    os.system(f'cd "{data_dir}" && {command}')
        else:
            print(f"Dataset {name} not found in the dataset dictionary.")

if __name__ == '__main__':
    install(["JRAIGS"], reinstall_all = False)
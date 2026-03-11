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
            "rm glaucoma-dataset.zip",
        ]
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
    
    if NAME_LIST == None:
        for key in DL_DATASET_DICT:
            if DL_DATASET_DICT[key].get("type", None) == "cli":
                for command in DL_DATASET_DICT[key]["commands"]:
                    run_in_data_dir(command)
    else:
        for name in NAME_LIST:
            if name in DL_DATASET_DICT:
                if DL_DATASET_DICT[name].get("type", None) == "cli":
                    for command in DL_DATASET_DICT[name]["commands"]:
                        run_in_data_dir(command)
            else:
                print(f"Dataset {name} not found in the dataset dictionary.")

if __name__ == '__main__':
    install()
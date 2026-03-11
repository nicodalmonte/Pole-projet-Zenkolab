from torchvision import transforms

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize(size=(224,224)), #dino richiede 224x224
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(), #Formato: tensore PyTorch Shape: [3, 224, 224]   ← (canali, altezza, larghezza) Valori: float da 0.0 a 1.0  (divide automaticamente per 255)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])
def get_val_transforms(): #da usare sia per evaluation set che per test set
    return transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



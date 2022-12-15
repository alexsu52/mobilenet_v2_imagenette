import sys

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from fastdownload import FastDownload
from sklearn.metrics import accuracy_score
from tqdm import tqdm

DEFAULT_CHECKPOINT = 'pytorch_model.bin'
DATASET_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'
DATASET_PATH = '~/.cache/nncf/datasets'
DATASET_CLASSES = 10


def load_checkpoint(model, path):  
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def validate(model, val_loader):
    predictions = []
    references = []
    
    with torch.no_grad():
        for images, target in tqdm(val_loader):
            output = model(images)
    
            predictions.append(np.argmax(output, axis=1))
            references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)  

    return accuracy_score(predictions, references)


downloader = FastDownload(base=DATASET_PATH, archive='downloaded', data='extracted')
path = downloader.get(DATASET_URL)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
val_dataset = datasets.ImageFolder(
    root=str(path / 'val'),
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    )
val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, num_workers=4, shuffle=False)

model = models.mobilenet_v2(num_classes=DATASET_CLASSES) 
model.eval()

checkpoint_path = DEFAULT_CHECKPOINT
if len(sys.argv) > 1:
    checkpoint_path = sys.argv[1]
model = load_checkpoint(model, checkpoint_path)

top1 = validate(model, val_loader)

print(f'Accuracy @ top1: {top1}')

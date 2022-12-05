import os
import sys

import torch
import torchvision.models as models

DATASET_CLASSES = 10

model = models.mobilenet_v2(num_classes=DATASET_CLASSES)    

if len(sys.argv) > 1:
    checkpoint_path = sys.argv[1]

    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])

        print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint_path, checkpoint['epoch']))

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "mobilenet_v2_imagenette.onnx", verbose=True)

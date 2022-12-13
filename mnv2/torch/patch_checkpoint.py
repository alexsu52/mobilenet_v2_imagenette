import os
import sys

import torch


def fix_names(state_dict):
    state_dict = {key.replace('module.', ''): value for (key, value) in state_dict.items()}
    return state_dict


checkpoint_path = sys.argv[1]
if os.path.isfile(checkpoint_path):
    print("=> loading checkpoint '{}'".format(checkpoint_path))

checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
checkpoint['state_dict'] = fix_names(checkpoint['state_dict'])
torch.save(checkpoint, 'pytorch_model.bin')
print('Patched checkpoint: pytorch_model.bin') 

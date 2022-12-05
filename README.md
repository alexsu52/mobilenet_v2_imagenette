# MobileNet v2 PyTroch on Imagenette
Training of MobileNet v2 from Torchvision on Imagenette dataset

## Run training:
```
python train.py --epochs 100 -j 6 -b 32 --lr 0.001 --wd 10e-5
```

## Patch checkpoint:
If you used DataParallel to train a model, the checkpoint has to be patched by removing 'module.' from keys of weights.
```
python patch_checkpoint.py <checkpoint>
```

## Pretrained model:
You can download pretrained model from [here](https://huggingface.co/alexsu52/mobilenet_v2_imagenette). Top-1 accuracy is 84.35%.

## Export to ONNX:
```
python export.py model_best.pth.tar
```

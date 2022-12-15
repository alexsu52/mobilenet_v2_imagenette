# MobileNet v2 TensorFlow on Imagenette
Training of MobileNet v2 from TensorFlow Keras Application on Imagenette dataset

## Run training:
```
python train.py --epochs 100 -b 32 --lr 0.001 --wd 10e-5 --pretrained
```

## Pretrained model:
You can download pretrained model from [here](https://huggingface.co/alexsu52/mobilenet_v2_imagenette). Top-1 accuracy is 98.77%.

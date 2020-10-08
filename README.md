# Pixel Privacy 2020

- Task description: https://multimediaeval.github.io/editions/2020/tasks/pixelprivacy/
- Koniq-10k dataset: http://database.mmsp-kn.de/koniq-10k-database.html

## File Structure:

```
this repo
│   train.py
│  
└───datasets  
│   │
│   └───koniq
│       │
│       └───pp2020_test
│       │
│       └───pp2020_dev
```


## Train full model:

```
!python train.py --path='/content/main/datasets/koniq' --batch_size=4 --num_epochs=100 --resume='/content/drive/My Drive/weights/Pixel Privacy/unet_ssim/unet_ssim_6_    0.8884.pth'
```

## Inference: #TODO
```
```

## References:
- SSIM Loss: https://github.com/Po-Hsun-Su/pytorch-ssim
- U Net: https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets

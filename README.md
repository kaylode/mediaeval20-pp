# Pixel Privacy 2020

- Task description: https://multimediaeval.github.io/editions/2020/tasks/pixelprivacy/
- Koniq-10k dataset: http://database.mmsp-kn.de/koniq-10k-database.html

# File Structure:

```
this repo
│   train.py
│   eval.py
│   fgsm.py
│  
└───datasets  
│   │
│   └───koniq
│       │
│       └───pp2020_test
│       │
│       └───pp2020_dev
│       │
│       └───enhance
│           │
│           └───pp2020_test
│           │
│           └───pp2020_dev
```

# Method:

<img src="./images/pipeline.PNG" width="700">

# Dataset:
- Pixel privacy 2020 dataset (Koniq-10k): [link](https://drive.google.com/file/d/1aYyZW4bcGSsRouRuo4HrNg37wFJQp1Bx/view?usp=sharing)
- Enhanced labels (of above dataset): [link](https://drive.google.com/file/d/1BefYNHFxFim5tT_V7dP5Cxo8-eDrZNlU/view?usp=sharing)

# Pretrained weights:
- BIQA checkpoint: [link](https://drive.google.com/file/d/1t8nOxtM4tQhOOQZmYZ1O1ltbywLSAaXe/view?usp=sharing)

# Image-to-Image:

## Train full U-Net model:
```
python train.py --path=<path to dataset> --batch_size=<size> --num_epochs=<epochs> --resume=<path to checkpoint>
```
## Inference using U-Net:
```
python eval.py --images=<input path to image folder:output path> --pretrained=<path to trained network weight>
```

# Two-stage approaches:
- ***Using FGSM***
```
python fgsm.py  --config=<path to yaml config> \
                --images=<input path to image folder:output path> \ 
                --enhance=<path to enhanced labels>
```

- ***To enhance image using PIL.ImageEnhance***
```
cd utils
python enhance.py  --images=<input path to image folder:output path>
```

# Results on Mediaeval 2020:
Methods | Accuracy (after JPEG 90) | Number of times selected as “Best” 
--- | --- | ---
***HCMUS_Team_run1_enlightment_gan_JPEG90*** | ***0.00*** | ***84***
HCMUS_Team_run3_unet_mse_enhance_JPEG90 | 13.27 | 11
HCMUS_Team_run2_fgsm_pillow_JPEG90 | 48.18 | 60
HCMUS_Team_run4_cartoon_attack_JPEG90 | 1.27 | 34
HCMUS_Team_run5_retouch_JPEG90 | 0.18 | 34


# References:
- SSIM Loss: https://github.com/Po-Hsun-Su/pytorch-ssim
- U Net: https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets
- FGSM: https://savan77.github.io/blog/imagenet_adv_examples.html

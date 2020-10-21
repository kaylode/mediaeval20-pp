import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageEnhance



def enhance(image):
    color = 1.5
    contrast = 1.5
    sharpness = 3.0
    
    enh_col = ImageEnhance.Color(image)
    image_colored = enh_col.enhance(color)
    enh_con = ImageEnhance.Contrast(image_colored)
    image_contrasted = enh_con.enhance(contrast)
    enh_sha = ImageEnhance.Sharpness(image_contrasted)
    image_sharped = enh_sha.enhance(sharpness)
   
    return image_sharped


def create_enhance(in_path, out_path):
    paths = os.listdir(in_path)
    for path in tqdm(paths):
        img_path = os.path.join(in_path, path)
        new_path = os.path.join(out_path, path)
        image = Image.open(img_path)
        enhanced = enhance(image)
        enhanced.save(new_path)

def main(args):
    if args.images is not None:
        input_path, output_path = args.images.split(':')
        
        if not os.path.exists(output_path):
            os.mkdir(output_path)

    create_enhance(in_path = input_path, out_path= output_path)

if __name__ == '__main__':
    parser.add_argument('--images', type=str, 
                        help='path to image folder:path to output folder')
    
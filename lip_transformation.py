
                    ################################
                    # Lip transformation functions #
                    ################################
# Command 
#python lip_transformation.py --input_dir './data/samples/' --out_dir './data/samples_augment/'

###########################################################
# Imports

import numpy as np
import torch
import os 
import glob 
import argparse
import utility_functions.CLP_transform_funcs as CL
from PIL import Image
import dlib
import cv2
from pathlib import Path

#################################################################
# Description: Model applies CLP transformations to control images

def main():
    parser = argparse.ArgumentParser(description='Directories for CL transformation process.')
    parser.add_argument('--input_dir', type=str, help='Directory of the control images to be transformed.')
    parser.add_argument('--out_dir', type=str, default='./data/samples_augment/', help='Output directory for the CL transformed images.')

    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Directory '{args.out_dir}' created")
    
    predictor = dlib.shape_predictor('weights/dlib/shape_predictor_68_face_landmarks-fbdc2cb8.dat')
    face_detector = dlib.get_frontal_face_detector()

    # List of control images to be transformed
    list_imgs = sorted(glob.glob(os.path.join(args.input_dir, '*.[jpJP][pnPN]*[gG]')))

    # List of landmarks to be used as inspiration for CL transformation
    clp_lks = glob.glob('./data/CLP_inspo_lks/*')

    for filename in list_imgs:
        img_cntr = np.array(Image.open(filename))
        lk_cntr = CL.face_lks(filename, predictor, face_detector)
        out_img, warp_lm = CL.func_transform_cl(clp_lks, img_cntr, lk_cntr)
        pil_image = Image.fromarray(np.array(out_img))
        pil_image.save(args.out_dir+filename.split('/')[-1])
        np.save(args.out_dir+filename.split('/')[-1][:-3]+'npy', warp_lm)

    print('CL transformation process completed!')    

if __name__ == "__main__":
    main()
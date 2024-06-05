#########################################
#        Lip Normalization process      #
#########################################

# python lip_normalization.py --input_dir './data/samples/' --outdir_process_orig './data/samples_process/' --outdir_norm './data/samples_norm/' --del_temps True

# Libraries
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import dlib
import cv2
import numpy as np
import argparse
import glob 
import os 
from pathlib import Path
import shutil
import third_party.CodeFormer.scripts.crop_align_face_mod as cr
import third_party.CodeFormer.inference_codeformer_mod as restore
from third_party.CodeFormer.scripts.crop_align_face_mod import Image_Preprocess

def create_directory(path):
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"Directory '{path}' created successfully")
    except Exception as e:
        print(f"Error creating directory '{path}': {e}")

def face_lks(name, predictor, face_detector):
    frame = cv2.imread(name)
    faces = face_detector(frame)
    for face in faces:
        landmarks = predictor(frame, face)                
        points = []
        for i in range(0, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y  
            points.append((x, y))
        points = np.array(points, np.int32)
        return points


def main():
    parser = argparse.ArgumentParser(description='Directories for lip normalization process.')
    parser.add_argument('--input_dir', type=str, help='Directory of the control images to be transformed.')
    parser.add_argument('--outdir_process_orig', type=str, default='./data/samples_process/', help='Output directory for preprocessed original images.')
    parser.add_argument('--outdir_norm', type=str, default='./data/samples_norm/', help='Output directory for the restored images.')
    parser.add_argument('--del_temps', type=bool, default=True, help='Save directories for intermediate steps.')

    args = parser.parse_args()

    # Create directories
    
    temp_compres = '/'.join(args.outdir_process_orig.split('/')[:-2]) + '/temp_comp/'
    temp_compres_process = '/'.join(args.outdir_process_orig.split('/')[:-2]) + '/temp_comp_pro/'

    create_directory(temp_compres)
    create_directory(temp_compres_process)
    create_directory(args.outdir_process_orig)
    create_directory(args.outdir_norm)

    # Initialize ARCFace and swapper models
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)
    # Initialize image preprocessing
    im_prep = Image_Preprocess()
    # Initialize face landmark detector
    predictor = dlib.shape_predictor('weights/dlib/shape_predictor_68_face_landmarks-fbdc2cb8.dat')
    face_detector = dlib.get_frontal_face_detector()


    print('###########Preprocessing original images###########\n')
    # List files
    list_imgs = sorted(glob.glob(os.path.join(args.input_dir, '*.[jpJP][pnPN]*[gG]')))
    for filename in list_imgs:
        # Preprocess for rotation and save processed images
        _, _ = im_prep.align_face(filename, args.outdir_process_orig+filename.split('/')[-1])
        lks = face_lks(args.outdir_process_orig+filename.split('/')[-1], predictor, face_detector)
        np.save(args.outdir_process_orig+filename.split('/')[-1][:-3]+'npy', lks)


    print('###########Image compression-decompression stage started###########\n')
    # Compression decompression and images saving 
    for filename in list_imgs:
        img = cv2.imread(filename)
        faces = app.get(img)
        faces = sorted(faces, key = lambda x : x.bbox[0])
        if len(faces)!= 0:
            source_face = faces[0]
            res = img.copy()
            for face in faces:
                res = swapper.get(res, face, source_face, paste_back=True)   
                cv2.imwrite(temp_compres+filename.split('/')[-1], res)
                try: 
                    _, _ = im_prep.align_face(temp_compres+filename.split('/')[-1], temp_compres_process+filename.split('/')[-1])
                except:
                    pass
        else:
            pass
    
    print('###########Image restoration stage started###########\n')
    restore.restore_run(temp_compres_process, args.outdir_norm, 0.5)

    for filename in sorted(glob.glob(os.path.join(args.outdir_norm, '*.[jpJP][pnPN]*[gG]'))):
        lks = face_lks(filename, predictor, face_detector)
        np.save(args.outdir_norm+filename.split('/')[-1][:-3]+'npy', lks)

    # Remove directories of intermediate steps
    if args.del_temps:
        shutil.rmtree(temp_compres)
        shutil.rmtree(temp_compres_process)

    print('###########Lip normalization finished succesfully###########\n')  
if __name__ == "__main__":
    main()
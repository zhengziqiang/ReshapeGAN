import cv2
import dlib
from PIL import Image
from pylab import *
import glob
import os
detector = dlib.get_frontal_face_detector()
import numpy as np

import argparse
def parse_args():
    desc = "Detect facial landmark"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_path', type=str, default='./data/celeba/celeba_images',required=True, help='')
    parser.add_argument('--save_path', type=str, default='./data/celeba/celeba_land', help='path to save facial landmark')
    parser.add_argument('--model_path', type=str, default='./shape_predictor_68_face_landmarks.dat', help='path of dlib predictor')
    return parser.parse_args()

def main():
    args=parse_args()
    if args is None:
        exit()
    data_path=args.data_path
    save_path=args.save_path
    model_path=args.model_path
    landmark_predictor = dlib.shape_predictor(model_path)
    for files in glob.glob(os.path.join(data_path,"/*.jpg")):
        p,n=os.path.split(files)
        img = cv2.imread(files)
        faces = detector(img,1)
        zero = np.zeros([img.shape[0], img.shape[1], 3], np.uint8)
        if (len(faces) > 0):
            for k,d in enumerate(faces):
                cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),(255,255,255))
                shape = landmark_predictor(img,d)
                for i in range(68):
                    cv2.circle(zero, (shape.part(i).x, shape.part(i).y), 5, (255, 255, 255), -1, 8)
                # you could also change the size of the key points by changing 5 to other numbers
        cv2.imwrite(os.path.join(save_path,n),zero)

if __name__ == '__main__':
    main()
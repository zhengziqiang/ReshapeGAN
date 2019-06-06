import glob
import os
from PIL import Image
import numpy as np
import argparse
def parse_args():
    desc = "Generating combinations"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_path', type=str, default='./data/celeba/celeba_images',required=True, help='image path')
    parser.add_argument('--save_path', type=str, default='./data/celeba/combination', help='path to save image combiantions')
    parser.add_argument('--n_ref', type=int, default=5,help='how many ref images in one combination')
    return parser.parse_args()

def main():
    args=parse_args()
    if args is None:
        exit()
    n_ref=args.n_ref
    data_path=args.data_path
    save_path=args.save_path
    ref_images=glob.glob(os.path.join(data_path,"*.*"))
    if len(ref_images):
        np.random.shuffle(ref_images)
        ipt_image=Image.open(ref_images[0])
        ipt_crop=ipt_image.crop((0,0,256,256))
        for i in range(len(ref_images)//n_ref):
            target=Image.new("RGB",((n_ref*2+1)*256,256))
            target.paste(ipt_crop,(0,0,256,256))
            for j in range(n_ref):
                tmp=Image.open(ref_images[i*n_ref+j])
                target.paste(tmp,(256+512*j,0,256+512*j+512,256))
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            target.save(os.path.join(save_path,"combination_"+str(i).zfill(4)+".jpg"))
    else:
        print("The image path have no files")
        exit()
if __name__ == '__main__':
    main()
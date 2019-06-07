import os
from PIL import Image
import glob
from pylab import *
import numpy as np
import argparse
def parse_args():
    desc = "facial normalized"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--img_path', type=str, default='./data/celeba/celeba_images',required=True, help='')
    parser.add_argument('--pose_path', type=str, default='./data/celeba/celeba_land', help='path to save facial landmark')
    parser.add_argument('--merge_path', type=str, default='./data/celeba/celeba_merge', help='path to save merged images')
    parser.add_argument('--norm_path', type=str, default='./data/celeba/celeba_norm',help='path to save normed images')
    return parser.parse_args()
def main():
    args=parse_args()
    if args is None:
        exit()
    img_path=args.img_path
    pose_path=args.pose_path
    merge_path = args.merge_path
    norm_path = args.norm_path
    if not os.path.exists(merge_path):
        os.mkdir(merge_path)
    if not os.path.exists(norm_path):
        os.mkdir(norm_path)
    for files in glob.glob(os.path.join(pose_path,"*.jpg")):
        img=Image.open(files)
        width,height=img.size
        data=array(img)
        if min(width,height)<128 or (width*1.0/height)<0.8 or (width*1.0/height)>1.2 or np.max(data)<200:
            continue
        else:
            p,n=os.path.split(files)
            l=Image.open(os.path.join(img_path,n))
            l_r=l.resize((256,256),Image.ANTIALIAS)
            r_r=img.resize((256,256),Image.ANTIALIAS)
            target=Image.new("RGB",(512,256))
            target.paste(l_r,(0,0,256,256))
            target.paste(r_r, (256, 0, 512, 256))
            target.save(os.path.join(merge_path,n))

    for files in glob.glob(os.path.join(merge_path,"*.jpg")):
        img=Image.open(files)
        pose=img.crop((256,0,512,256))
        pose_gray=pose.convert("L")
        pose_data=array(pose_gray)
        pose_data = np.where(pose_data >=127, pose_data, 0)

        column_index=np.argmax(pose_data,axis=0)
        column_index[:5]=0
        column_index[-5:] = 0
        column_min=np.where(column_index>0)[0][0]
        column_max = np.where(column_index > 0)[-1][-1]

        row_index=np.argmax(pose_data,axis=1)
        row_index[:5]=0
        row_index[-5:] = 0
        row_min=np.where(row_index>0)[0][0]
        row_max = np.where(row_index > 0)[-1][-1]

        mid_row=(row_min+row_max)//2
        mid_col = (column_min + column_max) // 2

        gap_height=(mid_row-row_min)*1.2
        gap_width=(mid_col-column_min)*1.2

        scale=gap_height*1.0/gap_width

        if scale<0.8:
            gap_height=gap_width*0.8
        else:
            gap_height=gap_width
        if scale>1.2:
            gap_height=gap_width*1.2
        else:
            gap_height = gap_width
        norm_row_min=mid_row-int(gap_height)
        norm_row_max=mid_row+int(gap_height)
        norm_col_min = mid_col - int(gap_width)
        norm_col_max = mid_col + int(gap_width)
        if norm_row_min<0:
            norm_row_min=0
        if norm_row_max>255:
            norm_row_max=255

        if norm_col_min<0:
            norm_col_min=0
        if norm_col_max>255:
            norm_col_max=255

        crop_pose=pose.crop((norm_col_min,norm_row_min,norm_col_max,norm_row_max))
        crop_img=img.crop((norm_col_min,norm_row_min,norm_col_max,norm_row_max))
        crop_pose_resize=crop_pose.resize((256,256),Image.ANTIALIAS)
        crop_img_resize = crop_img.resize((256, 256), Image.ANTIALIAS)
        target=Image.new("RGB",(256,256))
        target.paste(crop_img_resize,(0,0,256,256))
        p,n=os.path.split(files)
        target.save(os.path.join(norm_path,n))

if __name__ == '__main__':
    main()
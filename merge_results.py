import os
import glob
from PIL import Image
import argparse
def parse_args():
    desc = "Merge results"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_path', type=str, default='./result/celeba',required=True, help='synthesized images path')
    parser.add_argument('--save_path', type=str, default='./result/celeba/merge', help='path to save merged image')
    return parser.parse_args()
def main():
    args=parse_args()
    if args is None:
        exit()
    cnt = 0
    data_path=args.data_path
    save_path=args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    ipt_img=[]
    for i in range(10):
        target=Image.new("RGB",(256*10+90,256*10+90),(255,255,255))
        top = 0
        down = 256
        for j in range(10):
            left=0
            right=256*5+40
            for k in range(2):
                img=Image.open(os.path.join(data_path,"images/combination_"+str(i*20+j*2+k).zfill(4)+"/"+"combination_"+str(i*20+j*2+k).zfill(4)+"_face.jpg"))
                ipt=img.crop((0,0,256,256))
                if cnt==0:
                    ipt_img.append(ipt)
                sub=Image.new("RGB",(256*5+40,256),(255,255,255))
                for item in range(5):
                    ref=img.crop(((item+1)*256,0,(item+2)*256,256))
                    gen=img.crop(((item+1+5)*256,0,(item+2+5)*256,256))
                    resize_ref=ref.resize((64,64),Image.ANTIALIAS)
                    box=Image.new("RGB",(68,68),(255,0,0))
                    box.paste(resize_ref,(2,2,66,66))
                    merge=Image.new("RGB",(256,256))
                    merge.paste(gen,(0,0,256,256))
                    merge.paste(box,(256-68,256-68,256,256))
                    sub.paste(merge,(266*item,0,266*item+256,256))
                    cnt+=1
                target.paste(sub,(left,top,right,down))
                left+=256*5+50
                right+=256*5+50
            top+=266
            down+=266
        enlarged_ipt=ipt_img[-1].resize((256*2+10,256*2+10))
        target.paste(enlarged_ipt,(266*4,266*4,266*4+522,266*4+522))
        target.save(os.path.join(save_path,"merge_"+str(i)+".jpg"))

if __name__ == '__main__':
    main()
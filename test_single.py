# coding=utf-8
import os
import glob
from utils import *
from ops import *
import tensorflow as tf
from model import StarGAN
import argparse
def parse_args():
    desc = "Tensorflow implementation of ReshapeGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='test', help='train or test ?')
    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=6, help='The number of resblock')
    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--n_ref', type=int, default=5, help='how many ref images in one combination')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    parser.add_argument('--input_img', type=str, default='./data/celeba/celeba_images/0000.jpg',
                        help='input image path')
    parser.add_argument('--ref_img', type=str, default='./data/celeba/celeba_images/0001.jpg',
                        help='reference image path')
    parser.add_argument('--result_img', type=str, default='./demo_celeba.jpg',
                        help='result image path')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/celeba',
                        help='Directory name to save the checkpoints')
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)
    return args


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    # open session
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = StarGAN(sess, args)
        # build graph
        gan.build_model(args)
        if args.phase == 'test':
            gan.test_single(args)
            print(" [*] Test finished!")
if __name__ == '__main__':
    main()
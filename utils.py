import scipy.misc
import numpy as np
import os
from scipy import misc
from scipy import io
import argparse
def load_test_data(image_path,n_critic, size=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size, size*(n_critic*2+1)])
    img = np.expand_dims(img, axis=0)
    img = normalize(img)
    concat_imgs=None
    landmark=None
    input_img=img[:,:,:size,:]
    for i in range(0,n_critic):
        fix_img=img[:,:,size+size*2*i:size*2*(i+1),:]
        fix_img = np.expand_dims(fix_img, axis=0)
        land_img = img[:, :, size+size + size * 2 * i:size * 2 * (i + 1)+size, :]
        land_img = np.expand_dims(land_img, axis=0)
        if i == 0:
            concat_imgs = fix_img
            landmark=land_img
        else:
            concat_imgs = np.concatenate([concat_imgs, fix_img], axis=0)
            landmark = np.concatenate([landmark, land_img], axis=0)
    return input_img,concat_imgs,landmark


def normalize(x):
    return x / 127.5 - 1


def Normalize(data):
    mx = np.max(data)
    mn = np.min(data)
    return (data - mn) / (mx - mn)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if size[0] != 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            for k in range(images.shape[3]):
                norm = Normalize(image[:, :, k])
                img[idx * h:idx * h + h, k * w:k * w + w] = norm

        return img
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img

    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img

    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def inverse_transform(images):
    return (images + 1.) / 2.


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')


def parse_args():
    desc = "Tensorflow implementation of StarGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='test', help='train or test ?')
    parser.add_argument('--dataset', type=str, default='celeba', help='dataset_name')
    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=6, help='The number of resblock')
    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    return check_args(parser.parse_args())
"""checking arguments"""


def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)
    # --result_dir
    check_folder(args.result_dir)
    return args


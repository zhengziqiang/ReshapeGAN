
## Introduction
ReshapeGAN is a [Tensorflow](http://tensorflow.org/)-based framework for training and testing of our paper **[ReshapeGAN: Object Reshaping by Providing A Single Reference Image](https://arxiv.org/pdf/1905.06514.pdf)**.

## Installation
1. We use [Miniconda3](https://conda.io/miniconda.html) as the basic environment. If you have installed the Miniconda3 in path `Conda_Path`, please install `tensorflow-gpu` using the command `Conda_Path/bin/conda install -c anaconda tensorflow-gpu==1.8`.
2. Install dependencies by `Conda_Path/bin/pip install -r requirements.txt` (if necessary). The `requirements.txt` file is provided in this package.

## Train
The training code will be released soon!

## Test
### CelebA
We provide the pre-trained model and 1000 testing images for evaluation. 
Pre-trained model celeba.zip ([Google Drive](https://drive.google.com/open?id=17WP7YEU3u1lC-Fl09GulLgW8CSoGAGGB) and [Baidu Drive](https://pan.baidu.com/s/1y5raFPz5cManQqJ2DRk_9A)). 
Testing images used in our paper: `celeba_images.zip`.

After download the pre-trained model and testing images, run
```
mkdir checkpoint
mv celeba.zip checkpoint/
cd checkpoint/
unzip celeba.zip
cd ..
mkdir -p data/celeba
mv celeba_images.zip data/celeba/
cd data/celeba
unzip celeba_images.zip
cd ../..
```

Then we provide two files: `test_single.py` and `test_multiple.py` for generating only one image and multiple images respectively. For generating one image:
```
Conda_path/bin/python test_single.py # generating one image at once
--input_img ./data/celeba/celeba_images/0000.jpg # input image, you could also change this to any input image (0001.jpg, 0002.jpg and etc)
--ref_img ./data/celeba/celeba_images/0001.jpg # reference image, you could also change this to any reference image (0001.jpg, 0002.jpg and etc)
--checkpoint_dir ./checkpoint/celeba # checkpoint path
--result_img ./demo_celeba.jpg # result image
```
For generating multiple images using different reference images:
```
Conda_path/bin/python generate_multiple.py # generate combinations of reference images, one combination has 5 different reference images, and we randomly use one image as input image.
--data_path ./data/celeba/celeba_images # images path for providing reference images
--save_path ./data/celeba/combination # path to save image combinations
--n_ref # how many reference images in one image combination, we set default is 5

Conda_path/bin/python test_multiple.py 
--input_dir ./data/celeba/combination # input image path
--checkpoint_dir ./checkpoint/celeba # checkpoint path
--result_dir ./result/celeba  # result path
```
To make it easier to show the synthesized images, we also provide the `merge_result.py` to merge the generated images to a larger image.
```
Conda_path/bin/python merge_results.py
--data_path ./result/celeba # the results path
--save_path ./result/celeba/merge # the path to save merged images. 
```
We merge 96 synthesized images to a large image, and the middle enlarged image is the input image.

### UTKFace
We provide the pre-trained model and 1000 testing images for evaluation. 
Pre-trained model utk.zip ([Google Drive](https://drive.google.com/open?id=1QMqpzXdDT7O1uKhvIfkMoDeDCoiOwz9V) and [Baidu Drive](https://pan.baidu.com/s/1f9UiOhlHdr9oSkbNAuZmgg)).
Testing images used in our paper: `utk_images.zip`.

After download the pre-trained model and testing images, run
```
mv utk.zip checkpoint/
cd checkpoint/
unzip utk.zip
cd ..
mkdir data/utk
mv utk_images.zip data/utk/
cd data/utk
unzip utk_images.zip
cd ../..
```

Similarly, for generating one image:
```
Conda_path/bin/python test_single.py # generating one image at once
--input_img ./data/utk/utk_images/0000.jpg # input image, you could also change this to any input image (0001.jpg, 0002.jpg and etc)
--ref_img ./data/utk/utk_images/0001.jpg # reference image, you could also change this to any reference image (0001.jpg, 0002.jpg and etc)
--checkpoint_dir ./checkpoint/utk # checkpoint path
--result_img ./demo_utk.jpg # result image
```
For generating multiple images using different reference images:
```
Conda_path/bin/python generate_multiple.py # generate combinations of reference images, one combination has 5 different reference images, and we randomly use one image as input image.
--data_path ./data/utk/utk_images # images path for providing reference images
--save_path ./data/utk/combination # path to save image combinations
--n_ref # how many reference images in one image combination, we set default is 5

Conda_path/bin/python test_multiple.py 
--input_dir ./data/utk/combination # input image path
--checkpoint_dir ./checkpoint/utk # checkpoint path
--result_dir ./result/utk  # result path
```
To make it easier to show the synthesized images, we also provide the `merge_result.py` to merge the generated images to a larger image.
```
Conda_path/bin/python merge_results.py
--data_path ./result/utk # the results path
--save_path ./result/utk/merge # the path to save merged images. 
```
We merge 96 synthesized images to a large image, and the middle enlarged image is the input image.

## Datasets
Reshaping datasets: 
- KDEF
- RaFD
- FEI
- CelebA
- UTKFace
- Yale
- WSEFEP
- ADFES
- IIIT-CFW
- PHOTO-SKETCH 
- CUHK
- Face Sketch database 

## Data preparation
### Reshaping by within-domain guidance with paired data
```
├── demo
   ├── train
       ├── 000001.jpg 
       ├── 000002.jpg
       └── ...
   ├── test
       ├── a.jpg (The test image that you want)
       ├── b.png
       └── ... 
```
### Reshaping by within-domain guidance with unpaired data
```
├── demo
   ├── train
       ├── 000001.jpg 
       ├── 000002.jpg
       └── ...
   ├── test
       ├── a.jpg (The test image that you want)
       ├── b.png
       └── ... 
```

### Reshaping by cross-domain guidance with unparied data
```
├── demo
   ├── train
       ├── 000001.jpg 
       ├── 000002.jpg
       └── ...
   ├── test
       ├── a.jpg (The test image that you want)
       ├── b.png
       └── ...
   ├── attribute.txt (For domain attribute information for training)
   ├── attribute_test.txt (For domain attribute information for training) 
```

## Losses
- `DRAGAN`: [Perturbed loss](https://arxiv.org/abs/1705.07215).
- `LSGAN`: [Least Square GAN](https://arxiv.org/abs/1703.07737).

## Tutorial
### Train
Codes will be released soon!

## ReshapeGAN settings

<div style="text-align: center" />
<img src="./figures/illustration_demo.jpg" style="max-width: 500px" />
</div>

## The detailed ReshapeGAN model
### Reshaping by within-domain guidance with paired data
<div style="text-align: center" />
<img src="./figures/a.jpg" style="max-width: 500px" />
</div>

### Reshaping by within-domain guidance with unpaired data
<div style="text-align: center" />
<img src="./figures/b.jpg" style="max-width: 500px" />
</div>

### Reshaping by cross-domain guidance with unpaired data
<div style="text-align: center" />
<img src="./figures/c.jpg" style="max-width: 500px" />
</div>

## Results

<div style="text-align: center" />
<img src="./figures/demo.jpg" style="max-width: 500px" />
</div>


### CelebA
#### The 5 × 5 outputs by our ReshapeGAN using random 5 images as both inputs and references on CelebA dataset. 
<div style="text-align: center" />
<img src="./figures/5_celeba_demo.jpg" style="max-width: 500px" />
</div>

#### Random 96 synthesized images by our ReshapeGAN from one input sample on CelebA dataset. The middle enlarged image shows the input.
<div style="text-align: center" />
<img src="./figures/celeba_demo.jpg" style="max-width: 500px" />
</div>


### UTK
#### Random 96 generated results by our ReshapeGAN from one input sample on UTKFace dataset. The middle enlarged image shows the input.
<div style="text-align: center" />
<img src="./figures/utk_demo.jpg" style="max-width: 500px" />
</div>

### cat head 
#### Random 96 synthesized images by our ReshapeGAN from one input sample on Cat head dataset. The middle enlarged image shows the input.
<div style="text-align: center" />
<img src="./figures/cat_demo.jpg" style="max-width: 500px" />
</div>

### Panama 
#### Random 96 synthesized images by our ReshapeGAN from one input sample on Panama dataset. The middle enlarged image shows the input.
<div style="text-align: center" />
<img src="./figures/pose.jpg" style="max-width: 500px" />
</div>

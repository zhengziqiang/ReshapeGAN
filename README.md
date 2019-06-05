
## Introduction
ReshapeGAN is a [Tensorflow](http://tensorflow.org/)-based framework for training and testing of **[ReshapeGAN: Object Reshaping by Providing A Single Reference Image](https://arxiv.org/pdf/1905.06514.pdf)**

The training and testing codes will be released soon!

## Installation
1. We use [Miniconda3](https://conda.io/miniconda.html) as the basic environment. If you installed the Miniconda3 in path `Conda_Path`, please install `tensorflow-gpu` using the command `Conda_Path/bin/conda install -c anaconda tensorflow-gpu==1.8`.
2. Install dependencies by `Conda_Path/bin/pip install -r requirements.txt` (if necessary). The `requirements.txt` file is provided in this package.

## Train
The training code will be released soon!

## Test
### CelebA
We provide the pre-trained model and demo images for evaluation. 
Pretrained model ([Google Drive](https://drive.google.com/open?id=17WP7YEU3u1lC-Fl09GulLgW8CSoGAGGB) and [Baidu Drive](https://pan.baidu.com/s/1y5raFPz5cManQqJ2DRk_9A)). 
Testing images used in our paper: test.zip ([Google Drive](https://drive.google.com/open?id=1XGpbNROq3MsEQLv8FWWgs44XfPjAvFuP) and [Baidu Drive](https://pan.baidu.com/s/19uzVZnHaBPA4Ko6e0IWbIg)).
After download the pre-trained model and testing images.
```
unzip celeba.zip
mkdir -p checkpoint/CelebA
mkdir -p data/Celeba
mv celeba checkpoint/CelebA/pretrain
unzip test.zip
mv test/combination data/CelebA/combination
Conda_path/bin/python main.py --dataset ./data/CelebA --checkpoint_dir ./checkpoint/CelebA --result_dir ./result/celeba
```
You can test other demos by replacing the the first image of demo images with other input images. In theory, we can obtain 1000x1000 different images using 1000 images.

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

### Reshaping by cross-domain guidance
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


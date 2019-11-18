# JJ-style-transfer-TensorFlow2.0
Justin Johnson style transfer https://arxiv.org/abs/1603.08155 made using TensorFlow 2.0

## Usage 
To train a model by applying a style over content images, use this command. 
Dimensions (width x height) of both files should be the same.

`python train.py --content_img 'path_to_content_imgs/' --style_img 'path_to_style_img.png'`

To change some hyper parameters use the file `params.py`

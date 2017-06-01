# prisma
![Build Status](https://img.shields.io/teamcity/codebetter/bt428.svg)

ONLINE IMAGE STYLE TRANSFER

## Requirements

- Python 3+
- Tensorflow 1.1.0+
- Scipy
- Flask
- Flask-Mail
- Celery
- Redis

## Setup
- Dependencies
```
pip3 install numpy pillow scipy
pip3 install flask flask-mail celery redis
pip3 install tensorflow // for cpu
```
- Download Models
>http://pan.baidu.com/s/1pLPSXdx

```
mv models/ prisma/
```

- Mailbox
```
vim　default_config.py

MAIL_SERVER = 'xxxxx'
MAIL_PORT = xxx

MAIL_USERNAME = 'xxxxxx'
MAIL_PASSWORD = 'xxxxxx'
```

- Run Redis

- Run Celery
```
celery -A server.celery worker
```

- Run Flask
```
export FLASK_APP=server.py
flask run
```

## Training
- Download COCO dataset and VGG19 model
>[VGG19 model](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat)

>[COCO dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip)

- Put the model and dataset into "prisma/"

```
# Recommend using tensorflow of gpu version

python3 train.py --STYLE_IMAGES style-image.jpg --CONTENT_WEIGHT 1.0 --STYLE_WEIGHT 10.0 --MODEL_PATH models/newmodel.ckpt

mv models/newmodel.ckpt-done models/newmodel.ckpt

# Test
python3 eval.py --CONTENT_PATH content-image.jpg --MODEL_PATH models/newmodel.ckpt --OUTPUT_FOLDER output-images/
```

- Add to Flask app
```
vim default_config.py

MODEL_FILES = set(['newmodel.ckpt', ......])

# Put a sample image (Named "newmodel.jpg") into "static/models_image/"
```

## Screenshot
>http://localhost:5000/

![](https://github.com/hijkzzz/image-style-transfer/blob/master/screenshot.jpeg?raw=true)

## Reference
- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
- [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
- [OlavHN/fast-neural-style](https://github.com/OlavHN/fast-neural-style)

# PyTorch-YOLOv3-deepso
A minimal PyTorch implementation of YOLOv3, with support for training, inference and evaluation. You can get the additional information from original repo: https://github.com/eriklindernoren/PyTorch-YOLOv3


## Installation

##### Clone and install requirements
    $ git clone https://github.com/mozzi7428/PyTorch-YOLOv3.git
    $ cd PyTorch-YOLOv3/
    $ sudo pip3 install -r requirements.txt


## Preparation for deepso


#### Custom model
Run the commands below to create a custom model definition, replacing `<num-classes>` with the number of classes in your dataset.

```
$ cd config/                                # Navigate to config dir
$ bash create_custom_model.sh <num-classes> # Will create custom model 'yolov3-custom.cfg'
```
For deepso, since we have only one class, run
```
$ bash create_custom_model.sh 1
```

#### Classes
Add class names to `data/custom/classes.names`. This file should have one row per class name.

#### Dataset
Move deepso dataset to 'data/custom/'. Before running 'deepso2yolo.py', make sure that 1) All the images are placed in a form of 'data/pedestrian/image/[vidio_name]/frames_XX.jpg'. 2) All the annotation files are placed in a form of 'data/pedestrian/label/[vidio_name]-labels.json'. If the dataset is ready, run 'deepso2yolo.py'. This single line will: 
1) Rename and "move"(not copying) all the images to 'data/custom/images/'. 
2) Reform and copy all the annotations to 'data/custom/labels/'. Information of 'data/custom/images/[image_name].jpg' will be stored in 'data/custom/labels/[image_name].txt'. Each row in the annotation file should define one bounding box, using the syntax `label_idx x_center y_center width height`. The coordinates should be scaled `[0, 1]`, and the `label_idx` should be zero-indexed and correspond to the row number of the class name in `data/custom/classes.names`.
3) Add paths to images in 'data/custom/train.txt' and 'data/custom/valid.txt'

```
$ cd data/custom/
$ python deepso2yolo.py
```

#### Test
Evaluates the pretrained model on validation set. You can download pretrained model (img_size = 256) from here : https://drive.google.com/file/d/1eB3qwLosiLGO35_cFw9S6UIh_wPtpwqJ/view?usp=sharing and the model expect the pretrained weight is stored in 'weights/deepso.pth'.
```
$ python3 test.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --img_size 256 --weights_path weights/deepso.pth
```

#### Train
```
$ train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--gradient_accumulations GRADIENT_ACCUMULATIONS]
                [--model_def MODEL_DEF] [--data_config DATA_CONFIG]
                [--pretrained_weights PRETRAINED_WEIGHTS] [--n_cpu N_CPU]
                [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--evaluation_interval EVALUATION_INTERVAL]
                [--compute_map COMPUTE_MAP]
                [--multiscale_training MULTISCALE_TRAINING]
```

#### Example (deepso)
To train on custom images using a Darknet-53 backend pretrained on deepso images run: 
```
$ python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --img_size 256 --pretrained_weights weights/deepso.pth
```

#### Training log
```
---- [Epoch 7/100, Batch 7300/14658] ----
+------------+--------------+--------------+--------------+
| Metrics    | YOLO Layer 0 | YOLO Layer 1 | YOLO Layer 2 |
+------------+--------------+--------------+--------------+
| grid_size  | 16           | 32           | 64           |
| loss       | 1.554926     | 1.446884     | 1.427585     |
| x          | 0.028157     | 0.044483     | 0.051159     |
| y          | 0.040524     | 0.035687     | 0.046307     |
| w          | 0.078980     | 0.066310     | 0.027984     |
| h          | 0.133414     | 0.094540     | 0.037121     |
| conf       | 1.234448     | 1.165665     | 1.223495     |
| cls        | 0.039402     | 0.040198     | 0.041520     |
| cls_acc    | 44.44%       | 43.59%       | 32.50%       |
| recall50   | 0.361111     | 0.384615     | 0.300000     |
| recall75   | 0.222222     | 0.282051     | 0.300000     |
| precision  | 0.520000     | 0.300000     | 0.070175     |
| conf_obj   | 0.599058     | 0.622685     | 0.651472     |
| conf_noobj | 0.003778     | 0.004039     | 0.004044     |
+------------+--------------+--------------+--------------+
Total Loss 4.429395
---- ETA 0:35:48.821929
```

#### Tensorboard
Track training progress in Tensorboard:
* Initialize training
* Run the command below
* Go to http://localhost:6006/

```
$ tensorboard --logdir='logs' --port=6006
```

## Credit

### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```

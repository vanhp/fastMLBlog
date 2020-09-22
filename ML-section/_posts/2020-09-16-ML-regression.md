---
toc: false
layout: post
description: Info about ML-deep learning
category: ML
title: Regression
image: /ML-section/assets/images/ai-pic2.jpg
---
<!-- ![]({{page.image | relative_url}}) -->
{% include alert.html text="Warning: This page is under heavy construction and constant changing" %}
---
Dealing with value that are continuous instead of discrete which the realm of classification.
Image regression is area of application that use image as independent variable (x) and floating object(continous value) on the image as dependent variable. Which can be treat as another CNN on top of data block API.


Data set use:
 the [Biwi Kinect Head Pose](https://www.kaggle.com/kmader/biwi-kinect-head-pose-database) dataset 




### DataBlock

code for image regression

```
img_files = get_image_files(path)
def img2pose(x): return Path(f'{str(x)[:-7]}pose.txt')
img2pose(img_files[0])

im = PILImage.create(img_files[0])
im.shape
im.to_thumb(160)

# extract the head center point:
cal = np.genfromtxt(path/'01'/'rgb.cal', skip_footer=6)
def get_ctr(f):
    ctr = np.genfromtxt(img2pose(f), skip_header=3)
    c1 = ctr[0] * cal[0][0]/ctr[2] + cal[0][2]
    c2 = ctr[1] * cal[1][1]/ctr[2] + cal[1][2]
    return tensor([c1,c2])

get_ctr(img_files[0])
# get_y, since it is responsible for labeling each item.
biwi = DataBlock(
    # do image regression with x,y
    blocks=(ImageBlock, PointBlock),
    get_items=get_image_files,
    get_y=get_ctr,
    # make sure validation set contain 1 or more person not in train set
    splitter=FuncSplitter(lambda o: o.parent.name=='13'),
    batch_tfms=[*aug_transforms(size=(240,320)), 
                Normalize.from_stats(*imagenet_stats)]
)
dls = biwi.dataloaders(path)
dls.show_batch(max_n=9, figsize=(8,6))
# check the data
xb,yb = dls.one_batch()
xb.shape,yb.shape
yb[0]
learn = cnn_learner(dls, resnet18, y_range=(-1,1))
# define the range of dependent data that is expected in the data set
# since PyTorch and fastai treat left bottom is -1,top/right +1
def sigmoid_range(x, lo, hi): return torch.sigmoid(x) * (hi-lo) + lo
plot_function(partial(sigmoid_range,lo=-1,hi=1), min=-4, max=4)
# use default loss function
dls.loss_func
learn.lr_find()
# try this value
lr = 1e-2
learn.fine_tune(3, lr)
learn.show_results(ds_idx=1, nrows=3, figsize=(6,8))


```
MSELoss is used trying to predict something as close as possible



*Note*
- random splitter is not applicable since same person appear in multiple images
- each folder in the dataset contain image of one person
-  create a splitter that return true for a person a validation set for just that person
- second block is a Pointblock to let fastai know that the label represent coordinates when do augmentation
it would apply the same to the image folder

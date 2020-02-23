import os
import sys
import cv2
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import random

from mrcnn import utils
import mrcnn.model as modellib

import coco as coco

if not os.path.join("mask_rcnn_coco.h5"):
    utils.download_trained_weights("mask_rcnn_coco.h5")


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                    'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                    'teddy bear', 'hair drier', 'toothbrush']

def drawbb(img,clas,coordinates):
    y1,x1,y2,x2=coordinates
    color=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
    cv2.rectangle(img,(x1,y1),(x2,y2), color, 2)
    scale = 0.05
    fontScale = min(img.shape[1],img.shape[0])/(25/scale)
    cv2.putText(img, class_names[clas], (x1,y1), cv2.FONT_HERSHEY_COMPLEX, fontScale, color, 2, cv2.LINE_AA)
    return img


def drawbbs(img, rois, class_ids):
    n=len(class_ids)
    for i in range(n):
        img=drawbb(img, class_ids[i],rois[i])
    return img


class bb:
    def __init__(self):
        config = InferenceConfig()
        config.display()
        self.model = modellib.MaskRCNN(mode="inference", model_dir="logs", config=config)

        self.model.load_weights("mask_rcnn_coco.h5", by_name=True)


        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                    'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                    'teddy bear', 'hair drier', 'toothbrush']

    def predict_img(self,img):
        result=self.model.detect([img], verbose=0)
        r=result[0]

        return drawbbs(img, r['rois'], r['class_ids'])

    def predict_vid(self,vid):
        self.result=model.detect(vid, verbose=0)
        self.r=self.result[0]
        for i in range(vid.shape[0]):
            a=list()
            a.append(drawbbs(vid[i], self.r['rois'], self.r['class_ids']))
        return np.asarray(a)

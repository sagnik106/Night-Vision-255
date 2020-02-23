import rcnn
import cv2
import requests
import json
import urllib.request
import os
import numpy as np
b=rcnn.bb()
def boundb(img):
    f=b.predict_img(img)
    return f

def colorize(img):
    cv2.imwrite("temp_img.jpg",img)
    path=str(os.getcwd())+'/temp_img.jpg'
    r = requests.post(
        "https://api.deepai.org/api/colorizer",
        files={
            'image': open(path, 'rb'),
        },
        headers={'api-key': '364a966f-7800-4738-a13c-1ed493801d0e'}
    )
    url=r.json()['output_url']
    urllib.request.urlretrieve(url,filename='temp_img.jpg')
    return cv2.imread("temp_img.jpg")

def process(img):
    a=colorize(img)
    return boundb(a)

def process_vid(vid):
    a=list()
    for i in range(vid.shape[0]):
        a.append(process(vid[i]))
    return np.asarray(a)
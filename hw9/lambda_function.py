#!/usr/bin/env python
# coding: utf-8

import numpy as np
#import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path='dino-vs-dragon-v2.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


from io import BytesIO
from urllib import request

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


#url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Smaug_par_David_Demaret.jpg/1280px-Smaug_par_David_Demaret.jpg'

def preprocess(url):
    img = download_image(url)
    img = prepare_image(img, (150,150))
    img = np.array(img)
    img = img*(1./255)
    img = np.float32(img)
    X = np.array([img])
    return X

def predict(url):
    X = preprocess(url)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    float_preds = preds[0].tolist()
    return float_preds

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result



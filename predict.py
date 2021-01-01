#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:45:05 2020

@author: vidya sagar
"""

import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class pneumonia:
    def __init__(self,filename):
        self.filename =filename


    def predictionpneumonia(self):
        # load model
        model = load_model('pneumonia.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        if result[0][0] < 0.5:
            prediction = 'Normal'
            return [{ "image" : prediction}]
        else:
            prediction = 'Pneumonia'
            return [{ "image" : prediction}]



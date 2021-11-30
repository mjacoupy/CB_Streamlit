#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:39:04 2021

@author: maximejacoupy
"""

from deepface import DeepFace
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


data = "/Users/maximejacoupy/Downloads/Odile.jpeg"

pil_image = Image.open(data).convert('RGB')
open_cv_image = np.array(pil_image)
image = cv2.cvtColor(open_cv_image[:, :, ::-1], cv2.COLOR_BGR2RGB)

analyse = DeepFace.analyze(image, actions=['age', 'gender', "emotion", "race"])
age = analyse['age']
genre = analyse['gender']
emotion = analyse['dominant_emotion']
race = analyse['dominant_race']
    
plt.imshow(image)    
print(age, genre, emotion, race)    


# Authors: Jack Volgren
# Date: Jun 15 2023
# Purpose: A class to access the functionality of a trained neural network that gives steering commands.

import tensorflow as tf
import numpy as np
import cv2 as cv
from pathlib import Path

class DLAngular:

    def __init__(self, model_path, std, mean):
        self.model = tf.keras.models.load_model(model_path)
        self.dampening_factor = 0.8
        self.last_guess = None
        self.std = std
        self.mean = mean
        self.unstandard = lambda x: (x + self.mean) * self.std

    def dampen(self, previous_angle, new_angle):
        """
        Dampening function that slows down dramatic changes in inputs.
        
        Args:
            current_value (float): The current value.
            new_value (float): The new value to be applied.
            dampening_factor (float): The dampening factor (between 0 and 1).

        Returns:
            float: The dampened value.
        """
        
        dampened_value = previous_angle + (new_angle - previous_angle) * self.dampening_factor

        return dampened_value
    
    def predict(self, img):
       
        img = self.process(img)

        prediction = self.model.predict(img)[0][0]

        #if self.last_guess != None: prediction = self.dampen(self.last_guess, prediction)
        #else: self.last_guess = prediction

        prediction = (prediction + self.mean) * self.std

        return prediction
    
    def process(self, img):
        #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #img = cv.equalizeHist(img)
        #img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        #_, img = cv.threshold(img, 250, 255, cv.THRESH_BINARY)
        img = img[0:][720:]
        img = cv.resize(img, (100, 100))
        img = np.expand_dims(img, axis=0)
        return img

if __name__ == "__main__":
    DLAM = DLAngular("../angular_model", -0.0007405260112136602, -0.16777582466602325)
    all_images = list(Path("images").glob("*.png"))
    all_images = [str(x) for x in all_images]
    for i in all_images:
        img = cv.imread(i)
        pred = str(DLAM.predict(img)[0][0])
        
        #img = cv.putText(img, pred, (50,50), cv.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 3)
        #img = cv.resize(img, (900, 700))
        #cv.imshow("wind", img)
        img = np.squeeze(DLAM.process(img))
        cv.imshow("CV", img)
        cv.waitKey(0)

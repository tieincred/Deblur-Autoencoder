import numpy
import cv2
# import matplotlib.pyplot as plt
import tensorflow as tf
from utils.getBlurredArea import getBlurredROI
from utils.resizeSharpner import utilityFunctions
from utils.super_resolution import SuperResolution

class Inference:
    def __init__(self,model, superRes = False):
        self.model = model
        self.utility = utilityFunctions(intensity=1)
        self.super = SuperResolution()
        self.superRes = superRes


    def getResult(self, image):
        height, width, _ = image.shape
        if self.superRes:
            image = self.super.getSuperResolutionImage(image)
        result = self.utility.sharpenImage(image)
        image = cv2.resize(image,(128,128))
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255
        x_inp=image.reshape(1,128,128,3)
        result = self.model.predict(x_inp)
        result = result.reshape(128,128,3)
        result = result * 255
        # plt.imshow(result)
        im_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        resizedimg = cv2.resize(im_rgb, (width, height))

        return resizedimg

    def getFinalResults(self,image):
        getblur = getBlurredROI()
        croppedImg,pt1coo,pt2coo = getblur.getBlurROI(image)
        result = self.getResult(croppedImg)
        result = self.utility.sharpenImage(result)
        x1,y1 = pt1coo
        x2,y2 = pt2coo
        final = image.copy()
        final[y1:y2, x1:x2] = result

        return final, image


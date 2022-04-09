import cv2
import numpy as np


class SuperResolution:
    def __init__(self,type="edsr"):
        self.type = type

    
    def getSuperResolutionImage(self,image):

        #FSRCNN

        sr = cv2.dnn_superres.DnnSuperResImpl_create()

        if self.type == "fsrcnn":
            path = "modelWeights/FSRCNN_x4.pb"
            sr.readModel(path)
            sr.setModel("fsrcnn",4)
            result = sr.upsample(image)
        
        #EDSR
        elif self.type == "edsr":
            path = "modelWeights/EDSR_x4.pb"
            sr.readModel(path)
            sr.setModel("edsr",4)
            result = sr.upsample(image)
        
        #LapSRN
        else:
            path = "modelWeights/LapSRN_x4.pb"
            sr.readModel(path)
            sr.setModel("lapsrn",4)
            result = sr.upsample(image)           

        print("result result shape", result.shape)
        print("real shape",image.shape)

        return result

import cv2
import os
from utils.getBlurredArea import getBlurredROI
from utils.resizeSharpner import utilityFunctions

# assign directory
directorylq = 'data/lq'
directoryhq = 'data/hq'
getBlurredROI = getBlurredROI()
resizer = utilityFunctions()
 
# iterate over files in
# that directory
for filename in os.listdir(directoryhq):
    fhq = os.path.join(directoryhq, filename)
    # fhq = os.path.join(directoryhq, filename)
    # checking if it is a file

    imghq = cv2.imread(fhq)
    # imghq = cv2.imread(fhq)
    # cv2.imshow('hq',imghq)
    blurredROIhq = getBlurredROI.getBlurROI(imghq)
    # blurredROIhq = getBlurredROI.getBlurROI(imghq)

    finalimghq = cv2.resize(blurredROIhq,(128,128))
    # finalimghq = resizer.image_resize(blurredROIhq)

    cv2.imwrite('ROIedHQ/'+str(filename), finalimghq)
    # cv2.imwrite('ROIedHQ/'+str(filename), finalimghq)

    cv2.waitKey(0)


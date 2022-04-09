import cv2
from utils.getBlurredArea import getBlurredROI
from utils.super_resolution import SuperResolution
from utils.resizeSharpner import utilityFunctions
from utils.autoencoder import autoencoder
from inference.inference import Inference

model = autoencoder()
autoencoderModel = model.load_model('modelWeights/autoencoder.h5')

inferencing = Inference(autoencoderModel,superRes=True)

processingType = 'image'

if processingType == 'image':
    paths = ['1_0001.png','1_0002.png','1_0003.png','1_0004.png','1_0005.png']
    for path in paths:
        image = cv2.imread(path)

        updated, original = inferencing.getFinalResults(image)

        cv2.imwrite('DemoAE/'+'Processed'+path,updated)

else:
    videoPath = ''
    cap = cv2. VideoCapture(videoPath)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('results.mp4', 
                            cv2.VideoWriter_fourcc(*'FMP4'),
                            10, size)
    while(cap. isOpened()):
        ret, frame = cap.read()
        try:
            frame_test = frame.copy()
        except:
            break
        
        updated, original = inferencing.getFinalResults(frame)
        result.write(updated)


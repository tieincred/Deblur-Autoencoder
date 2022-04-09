import cv2
import numpy as np
import mediapipe as mp

class getBlurredROI:
    def __init__(self, d=0):
        self.d = d
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()

    def getBlurROI(self,image):
        # print(image)
        height, width, _ = image.shape
        print("Height, width", height, width)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = self.face_mesh.process(rgb_image)
        for facial_landmarks in result.multi_face_landmarks:
            for i in range(0, 1):
                pt1 = facial_landmarks.landmark[50]
                pt2 = facial_landmarks.landmark[379]

        # cv2.circle(image, (x, y), 3, (100, 100, 0), -1)
        pt1coo = (int(pt1.x * width)-10,int(pt1.y * height))
        pt2coo = (int(pt2.x * width)+10,int(pt2.y * height))

        x1,y1 = pt1coo
        x2,y2 = pt2coo
        cropped_image = image[y1:y2, x1:x2]

        return cropped_image, (pt1coo), (pt2coo)
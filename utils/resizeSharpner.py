import cv2

class utilityFunctions:
    def __init__(self, intensity=2):
        self.intensity = intensity

    
    def sharpenImage(self,image):
        #sharpning the image
        gaussian_blur = cv2.GaussianBlur(image, (3,3), 2)

        if self.intensity == 1:
            sharpened = cv2.addWeighted(image,1.5, gaussian_blur, -0.5, 0)
        if self.intensity == 2:
            sharpened = cv2.addWeighted(image,7.5, gaussian_blur, -6.5, 0)
        else:
            sharpened = cv2.addWeighted(image,4.5, gaussian_blur, -3.5, 0)

        return sharpened


    def resizeToOriginal(self,original,imgToResize):
        # downscaling the image
        resizeFactor = original.shape[0]/imgToResize.shape[0]
        print('resizing with a factor of '+ str(resizeFactor))
        resizedimg = cv2.resize(imgToResize, (0, 0), fx=resizeFactor, fy=resizeFactor)
        
        return resizedimg

    def image_resize(self, image, width = 128, height = 128, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized


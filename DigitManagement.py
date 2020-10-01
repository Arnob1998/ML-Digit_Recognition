import os
from skimage import io
import numpy as np
from skimage.transform import resize



class DigitTransformClass:

    img_loc = None
    smoothening_with_AA = None

    def __init__(self, digit_location=None, anti_aliasing=True):
        if digit_location is None:
            self.img_loc = os.getcwd() + "\\screenshot.jpeg"
        else:
            self.img_loc = digit_location

        self.smoothening_with_AA = anti_aliasing
        print("\nWorking on the image")

    def toGrayScale(self, img):
        mean_O_rgb = np.mean(img, axis=2)
        return mean_O_rgb

    def transformPipeline(self):
        img_rgb1 = io.imread(self.img_loc)
        # img_grayscale1 = color.rgb2gray(img_rgb1) # this using different scaling as in 1 -> 255, so didn't use it
        img_grayscale1 = self.toGrayScale(img_rgb1)
        image_resized = resize(img_grayscale1, (28, 28), anti_aliasing=self.smoothening_with_AA)

        if self.smoothening_with_AA == False:
            print("\nLog : anti_aliasing is set to false. Digit is converted to binary")
            image_resized[image_resized > 0] = 255
            ready_img = image_resized.flatten()
        else:
            print("\nLog : anti_aliasing is set to true. Digit is converted to grayscale")
            ready_img = image_resized.flatten()

        return ready_img
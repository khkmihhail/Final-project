import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract

# load image
img = cv2.imread('page_text.jpg', 0)
img2 = cv2.imread('page_text2.jpg', 0)


# process image
# TODO: improve image proccesing for pytesseract. 
blur = cv2.medianBlur(img2,1)
gaus = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,5)

# ------------------


canny = cv2.Canny(img2, 100, 200)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5)) 
dilation = cv2.dilate(canny,kernel,iterations = 5)


# find counters
derp, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
areaArray = []

# getting biggest counter for smth...???
for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)

sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
bgst_cont = sorteddata[0][1]

# Draw rect
x, y, w, h = cv2.boundingRect(bgst_cont)
cv2.drawContours(dilation, bgst_cont, -1, (255, 0, 0), 2)
cv2.rectangle(gaus, (x, y), (x+w, y+h), (0,0,255), 2)
# Remove background and return only text area back/ crop image
text = gaus[y:y+h, x:x+w] # region of intrest. It`s working !!!

# read text
pytesseract.pytesseract.tesseract_cmd = 'C:\\Tesseract-OCR\\tesseract'
text_output = pytesseract.image_to_string(Image.fromarray(text), config='-psm 6')


# Show results
cv2.imshow('text', text)
print(text_output)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Useful links
##https://github.com/danvk/oldnyc/blob/master/ocr/tess/crop_morphology.py
##https://github.com/opencv/opencv_contrib/blob/master/modules/text/samples/textdetection.py
##http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
##https://mmeysenburg.github.io/image-processing/08-contours/

import cv2
import numpy as np

im = cv2.imread('/home/nightbreeze/deeplearning/oneposeplus/data/demo/chuizi7/chuizi7-test/affmask_image/197.png')
width = int(im.shape[1]/2)
height = int(im.shape[0]/2)
dim = (width , height)
im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('image',im)
cv2.waitKey()
cv2.destroyAllWindows()

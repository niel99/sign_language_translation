import cv2
import numpy as np
import copy
import math
import os
import tensorflow.keras as keras

folder = 'E'
if os.path.exists(folder) == False:
    os.mkdir(folder)
cap_region_x_begin=0.5  
cap_region_y_end=0.8  
threshold = 60  
blurValue = 25  
bgSubThreshold = 50
learningRate = 0
blur = []

idx = 0

model = keras.models.load_model("model")

# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False 

def printThreshold(thr):
    print("! Changed threshold to "+str(thr))


def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

def get_prediction(arr, model):
    classes = model.predict_classes(arr.reshape(1,28,28,1))
    if len(classes) > 0:
        return get_prediction(classes[0])
    return ''

camera = cv2.VideoCapture(0)
camera.set(10,200)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)
thresh = []

while camera.isOpened():
    ret, frame = camera.read()
    # print(frame.shape)
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        fgmask = bgModel.apply(frame,learningRate=learningRate)
        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        cv2.imshow('bgmask', fgmask)
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        cv2.imshow('mask', img)
        
        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('ori', thresh)




    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        camera.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print( '!!!Background Captured!!!')
    elif k == ord('r'):  # press 'r' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print ('!!!Reset BackGround!!!')
    elif k == ord('n'):
        triggerSwitch = True
        print ('!!!Trigger On!!!')
    elif k == ord('c'):
        print("!!! Image Captured!!!")
        cv2.imwrite('{}/captured{}.jpg'.format(folder,idx), blur)
        idx+=1
    elif k == ord('q'):
        print("!!!Quitting!!!")
        break

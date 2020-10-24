import cv2
import numpy as np 

# source: https://medium.com/analytics-vidhya/hand-detection-and-finger-counting-using-opencv-python-5b594704eb08

# vid = cv2.VideoCapture(0)
# while (True):
#     ret, frame = vid.read()
#     #Import image
#     frame = cv2.imread("palm_image.jpeg")
#     if ret == True:
#         hsvim = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         lower = np.array([0, 48, 80], dtype="uint8")
#         upper = np.array([20, 255, 255], dtype="uint8")
#         skinRegionHSV = cv2.inRange(hsvim, lower, upper)
#         blurred = cv2.blur(skinRegionHSV, (2, 2))
#         ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)
#         contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
#         contours = max(contours, key=lambda x: cv2.contourArea(x))
#         cv2.drawContours(frame, [contours], -1, (255, 255, 0), 2)
#         cv2.imshow("contours", frame)

#         hull = cv2.convexHull(contours, returnPoints=False)
#         defects = cv2.convexityDefects(contours, hull)

#         if defects is not None:
#             cnt = 0
#         for i in range(defects.shape[0]):  # calculate the angle
#             s, e, f, d = defects[i][0]
#             start = tuple(contours[s][0])
#             end = tuple(contours[e][0])
#             far = tuple(contours[f][0])
#             a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
#             b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
#             c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
#             angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  #cosine theorem
#         if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
#             cnt += 1
#             cv2.circle(frame, far, 4, [0, 0, 255], -1)
#         if cnt > 0:
#             cnt = cnt+1
#         cv2.putText(frame, str(cnt), (0, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
#         cv2.imshow('final_result', frame)
#         cv2.imshow("Thresh", thresh)
#     if (cv2.waitKey(20) == 27):
#         break

# vid.release()
# cv2.destroyAllWindows()



#Source: https://medium.com/swlh/roi-segmentation-contour-detection-and-image-thresholding-using-opencv-c0d2ea47b787

import math
import copy

cap_region_x_begin = 100
cap_region_y_end = 100
x = 0.5
y = 0.8
threshold = 60
blurValue = 7
bgSubThreshold = 50
learningRate = 0
isBgCapture = 0

def removeBG(frame):
    fgmark = bgModel.apply(frame, learningRate = learningRate)
    kernel = np.ones((3,3), np.uint8)
    fgmark = cv2.erode(fgmark, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask = fgmark)
    return res
    
camera = cv2.VideoCapture(0)
ret = camera.set(3, 1280)
ret = camera.set(4, 720)
while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100) #smoothening filter
    frame = cv2.flip(frame, 1) #flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
    (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2) #draw
    cv2.imshow("Original", frame) #Show Image

    #Main operation
    if isBgCapture == 1: #this part wont run until background captured
        img = removeBG(frame)
        img = img[0:int(y*frame.shape[0]),
                    int(x* frame.shape[1]):frame.shape[1]] #clip the ROI
        cv2.imshow("mark", img)

        #convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        cv2.imshow("blur", blur)

        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY) #thresholding the frame
        cv2.imshow("ori", thresh)

        #get the contours
        thresh1 = copy.deepcopy(thresh)
        img2, contours, hierarchy = cv2.findContours(
            thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0 :
            for i in range(length): #find the biggest contours (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i
                
            res = contours[ci]
            hull = cv2.convexHull(res) #apply convex hull technique
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2) #drawing contours
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3) #drawing convex hull

        cv2.imshow("Output", drawing)

    #press q to quit
    k = cv2.waitKey(10)
    if k == 27:
        camera.release()
        cv2.destroyAllWindows()
        break
    #press b to set background
    elif k == ord('b'):
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCapture = 1
        print("Background Captured")
    #press r to reset background
    elif k == ord('r'):
        bgModel = None
        isBgCapture = 0
        print("Reset Background")

import cv2
import numpy as np
import sys
import matcher
import time

def kaze_match(im1_path, im2_path):
    img1 = cv2.imread('original_image_left.jpeg')
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread('original_image_right.jpeg')
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

#sift = cv2.xfeatures2d.SIFT_create()

#kp1,des1 = sift.detectAndCompute(img1,None)
#kp2,des2 = sift.detectAndCompute(img2,None)
    detector = cv2.AKAZE_create()
    (kps1, descs1) = detector.detectAndCompute(gray1, None)
    (kps2, descs2) = detector.detectAndCompute(gray2, None)


    print("keypoints: {}, descriptors: {}".format(len(kps1), descs1.shape))
    print("keypoints: {}, descriptors: {}".format(len(kps2), descs2.shape))
    print(2+2)

    #cv2.imshow('original_image_left_keypoints',cv2.drawKeypoints(img_,kp1,None))


    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descs1,descs2, k=2)    # typo fixed

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kps1, img2, kps2, good[1:20], None, flags=2)
    cv2.imshow("AKAZE matching", im3)
    cv2.waitKey(0)

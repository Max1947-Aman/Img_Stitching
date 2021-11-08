import cv2
import numpy as np

dim=(1024,768)  
left=cv2.imread('left.jpeg',cv2.IMREAD_COLOR)     #read img
left=cv2.resize(left,dim,interpolation=cv2.INTER_AREA)   #In case the images or of uneven size


right=cv2.imread('right.jpeg',cv2.IMREAD_COLOR)
right=cv2.resize(right,dim,interpolation=cv2.INTER_AREA)


sift=cv2.xfeatures2d.SIFT_create() #SIFT, which stands for Scale Invariant Feature Transform, is a method for extracting feature vectors that describe local patches of an image. 


kp1, des1=sift.detectAndCompute(left,None)   #storing the keypoints for matching
kp2, des2=sift.detectAndCompute(right,None)


FLANN_INDEX_KDTREE=0    #Matching images using Flann Index KDTree algo
index_params=dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params=dict(checks=50)
match=cv2.FlannBasedMatcher(index_params, search_params)

#match=cv2.BFMatcher() ----> It didnt worked so i used FlannBased Matcher and knn Matcher
matches=match.knnMatch(des1,des2,k=2)
good=[]    
for m,n in matches:    
	if m.distance < 0.1*n.distance:
		good.append(m)


draw_params=dict(matchColor=(0,255,0),singlePointColor=None,flags=2)  
im3=cv2.drawMatches(left,kp1,right,kp2,good,None,**draw_params)      #Drawing parameters over matched keypoints which you will see in green color lines


images=[]   #list to store both images for stitching
images.append(left)
images.append(right)

stitcher=cv2.Stitcher.create()  #Using stitcher class and create function to stitch the images on basis of feature matching
ret,pano=stitcher.stitch(images) #stitching and creating panorama of both the images and storing in ret and pano

if ret==cv2.STITCHER_OK:   #checking if the image is stitched properly
    cv2.imshow('left_img',left)
    cv2.imshow('right_img',right)
    cv2.imshow('matching_params',im3)  #Displays the matching keypoins and prameters drawn over them
    cv2.imshow('stitched_img',pano)    #DIsplays Stitched image
    cv2.imwrite('original_image_drawmatches.jpeg',im3)  #saving the img
    cv2.imwrite("output.jpeg",pano)   #saving the img
    cv2.waitKey(50000)  #the output image will last for 20 sec(=20000) u can increse or decrease it by changing the figure inside waitKey
    cv2.destroyAllWindows()  #closes all window after 20 sec
else:
    print("Error During Stitching!")  #Error message

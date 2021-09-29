# Finding Best match of each query image comparing same kind of reference images

import cv2
import numpy as np 
import matplotlib.pyplot as plt
import glob

def detect_features(image):
    descriptor = cv2.ORB_create()
    (keypoints, features) = descriptor.detectAndCompute(image, None)
    
    return (keypoints, features)

def match_keypoints(f1, f2,arg = 0):
    
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    match_points = matcher.match(f1,f2)
    
    return match_points

def display(image,caption = ''):
    plt.figure(figsize = (5,10))
    plt.title(caption)
    plt.imshow(image)
    plt.show()
    
def FindAndMatchDescriptors(img1, img2):
    gray_R = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray_Q = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) 
    
    pointsR, featuresR = detect_features(gray_R)
    pointsQ, featuresQ = detect_features(gray_Q)
    
    matched_points = match_keypoints(featuresR, featuresQ)
    

    result = cv2.drawMatches(imageR,pointsR,imageQ,pointsQ,matched_points,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    matrix,mask = estimateTransformation(pointsR, pointsQ, featuresR, featuresQ, matched_points)
    result = cv2.warpPerspective(imageR, matrix, (imageQ.shape[1], imageQ.shape[0]))
    inliers = findInliers(mask)

    return (result,matrix,inliers,len(mask))

def estimateTransformation(p1, p2, f1, f2, matches):
    
    p1 = np.float32([kp.pt for kp in p1])
    p2 = np.float32([kp.pt for kp in p2])
    
    if len(matches) > 4:
        # construct the two sets of points
        pts1 = np.float32([p1[m.queryIdx] for m in matches])
        pts2 = np.float32([p2[m.trainIdx] for m in matches])
    
        (H, mask) = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5)
        return (H, mask)
    else:
        return None

def findInliers(mask):
    in_points = 0
    for i in range(len(mask)):
        if mask[i] == 1:
            in_points = in_points+1
    
    return in_points

def FeatureMatching(imageR,imageQ,resize = 0):
    
    if resize == 1:
        imageQ = cv2.resize(imageQ,(0,0),fx = 0.6, fy = 0.6)
    
    result, matrix,inliers, total_points = FindAndMatchDescriptors(imageR, imageQ)

    return inliers, total_points



if __name__ == '__main__':
    
    path_query = '/Users/eapplestroe/Feature Matching Dataset/book_covers/Query/*.jpg' 
    IPQ = glob.glob(path_query)
    IPQ = np.sort(IPQ)
    
    path_reference = '/Users/eapplestroe/Feature Matching Dataset/book_covers/Reference/*.jpg' 
    IPR = glob.glob(path_reference)
    IPR = np.sort(IPR)
    
    
    size_flag = 0
    counter = 0
    tp = 0
    fp = 0
    
    for image_q in IPQ:
        imageQ = cv2.imread(image_q)
        for image_r in IPR:
            imageR = cv2.imread(image_r)

            if imageQ.shape[0] > 2*imageR.shape[0] or imageQ.shape[1] > 2*imageR.shape[1]:
                size_flag = 1
                
            inliers, total_points = FeatureMatching(imageR,imageQ,size_flag)
            
            percentage = (inliers*100)/total_points
            
            if percentage > 25:
                q = image_q[len(image_q)-7:len(image_q)-4]
                r = image_r[len(image_r)-7:len(image_r)-4]
                print('Query '+q+' matched with reference '+r)
                if q == r:
                    counter = counter+1
                    tp = tp+1
                else:
                    fp = fp+1
                    
    matched = tp*100/(len(IPQ))
    print('Total Matches = ', counter)
    print('True Matches = ',tp)
    print('False Matches = ',fp)
    print('Macthing Percentage = ',matched)

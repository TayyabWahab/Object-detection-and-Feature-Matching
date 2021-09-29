import cv2
import numpy as np 
import matplotlib.pyplot as plt

def draw_outline(ref, query, model):
    h,w,c = ref.shape
    pts = np.float32([ [0,0],[0,h],[w,h],[w,0] ]).reshape(-1,1,2)
    corners = cv2.perspectiveTransform(pts,model)
    
    img = cv2.polylines(query,[np.int32(corners)],True,(0,255,0),3, cv2.LINE_AA)
    display(img,'Object Outlined')
    

def detect_features(image):
    descriptor = cv2.ORB_create()
    (keypoints, features) = descriptor.detectAndCompute(image, None)
    
    return (keypoints, features)

def match_keypoints(f1, f2,arg = 0):
    
    if arg == 0:
        matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    elif arg == 1:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif arg == 2:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    match_points = matcher.match(f1,f2)
    
    return match_points

def display(image,caption = ''):
    plt.figure(figsize = (10,20))
    plt.title(caption)
    plt.imshow(image)
    plt.show()
    
def FindAndMatchDescriptors(img1, img2, match_arg = 0):
    gray_R = cv2.cvtColor(imageR, cv2.COLOR_RGB2GRAY)
    gray_Q = cv2.cvtColor(imageQ, cv2.COLOR_RGB2GRAY) 
    
    pointsR, featuresR = detect_features(gray_R)
    pointsQ, featuresQ = detect_features(gray_Q)
    
    matched_points = match_keypoints(featuresR, featuresQ,match_arg)
    matched_points.sort(key=lambda x: x.distance, reverse=False)
    
    
    print('Matched Points: ', len(matched_points))
    result = cv2.drawMatches(imageR,pointsR,imageQ,pointsQ,matched_points,None)
    
    return (result,pointsR,pointsQ,featuresR,featuresQ, matched_points)

def estimateTransformation(p1, p2, f1, f2, matches):
    
    p1 = np.float32([kp.pt for kp in p1])
    p2 = np.float32([kp.pt for kp in p2])
    
    if len(matches) > 4:
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


if __name__ == '__main__':
    imageR = cv2.imread('/Users/eapplestroe/Downloads/A2_smvs/book_covers/Reference/003.jpg')
    imageQ = cv2.imread('/Users/eapplestroe/Downloads/A2_smvs/book_covers/Query/003.jpg')
    
    (result1,pointsR1,pointsQ1,featuresR1,featuresQ1,matched_points1) = FindAndMatchDescriptors(imageR, imageQ,0)
    display(result1, 'Matching Using L1 Normalization')
    
    (result2,pointsR2,pointsQ2,featuresR2,featuresQ2,matched_points2) = FindAndMatchDescriptors(imageR, imageQ,1)
    display(result2, 'Matching Using L2 Normalization')
    
    (result,pointsR,pointsQ,featuresR,featuresQ,matched_points) = FindAndMatchDescriptors(imageR, imageQ,2)
    display(result, 'Matching Using Hamming Distance')
    
    matrix,mask = estimateTransformation(pointsR, pointsQ, featuresR, featuresQ, matched_points)
    result = cv2.warpPerspective(imageR, matrix, (imageQ.shape[1], imageQ.shape[0]))
    display(result, 'Extracted Object')

    draw_outline(imageR.copy(),imageQ.copy(),matrix)

    matrix1,mask1 = estimateTransformation(pointsR1, pointsQ1, featuresR1, featuresQ1, matched_points1)
    inliers = findInliers(mask1)
    print('Inliers(TP) and Outliers(FP) using Norm1 are: ',inliers,len(mask1)-inliers,'respectively')

    matrix2,mask2 = estimateTransformation(pointsR2, pointsQ2, featuresR2, featuresQ2, matched_points2)
    inliers = findInliers(mask)
    print('Inliers(TP) and Outliers(FP) using Norm2 are: ',inliers,len(mask2)-inliers,'respectively')

    matrix,mask = estimateTransformation(pointsR, pointsQ, featuresR, featuresQ, matched_points)
    inliers = findInliers(mask)
    print('Inliers(TP) and Outliers(FP) using Hamming Distance are: ',inliers,len(mask)-inliers,'respectively')





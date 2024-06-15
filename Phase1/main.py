#!/usr/bin/evn python

import argparse
import cv2
import numpy as np
from util import save_image, plot_corners, draw_matches, select_random_points

def ANMS(image, corners:np.array, Nbest:int): 
    harris_scores = cv2.cornerHarris(image, blockSize=2, ksize=3, k=0.04)
    scores = [harris_scores[y, x] for x, y in corners]

    num_corners = len(corners)
    radii = np.full(num_corners, np.inf)
    for i in range(num_corners):
        for j in range(num_corners):
            if scores[j] > scores[j]:
                ED = np.linalg.norm(corners[j] - corners[i])
                radii[i] = min(radii[i], ED)
    
    sorted = np.argsort(radii)[::-1]
    best_indices = sorted[:Nbest]
    return [corners[i] for i in best_indices]

def getFeatureVector(image:np.array, corners:np.array):
    patchSize = 40
    padded = cv2.copyMakeBorder(image, int(patchSize/2), int(patchSize/2), int(patchSize/2), int(patchSize/2), cv2.BORDER_CONSTANT)
    descriptors = []
    for c in corners:
        x, y = c[0] + patchSize/2, c[1] + patchSize/2
        patch = padded[int(y - patchSize/2):int(y + patchSize/2), int(x - patchSize/2):int(x + patchSize/2)]
        if patch.shape == (40, 40):
            blurred_patch = cv2.GaussianBlur(patch, (3,3), 0)
            subsample = cv2.resize(blurred_patch, (8, 8))
            subsample = subsample.reshape((64, 1))
            standardized = (subsample - np.mean(subsample)) / np.std(subsample)
            descriptors.append(standardized)

    return descriptors

def matchFeatures(des1, des2, corn1, corn2, ratio=0.8):
    kps1 = []
    kps2 = []
    dmatch = []
    for i in range(0, len(des1)):
        best_matches = [np.inf, np.inf]  # (best_match, 2nd best match)
        match_idx = [0, 0]
        for j in range(0, len(des2)):
            ssd = np.sum((des1[i] - des2[j])**2)
            if ssd < best_matches[0]:
                best_matches[1] = best_matches[0]
                best_matches[0] = ssd
                match_idx[1] = match_idx[0]
                match_idx[0] = j
            
        if best_matches[0]/best_matches[1] <= ratio:
            kps1.append(corn1[i])  
            kps2.append(corn2[match_idx[0]])
            dmatch.append(cv2.DMatch(len(kps1) - 1, len(kps2) - 1, best_matches[0]))

    return np.asarray(kps1), np.asarray(kps2), dmatch

def find_inliers(H, kps1, kps2, dmatches, threshold=5.0):
    inliers = []
    for match in dmatches:
        pt1 = np.array([kps1[match.queryIdx][0], kps1[match.queryIdx][1], 1])
        pt2 = np.array([kps2[match.trainIdx][0], kps2[match.trainIdx][1], 1])
        projected_pt1 = np.dot(H, pt1)
        if (projected_pt1[2] != 0):
            projected_pt1 = projected_pt1 / projected_pt1[2]  ## normalize step
            ssd = np.sum((projected_pt1 - pt2)**2)
            if ssd < threshold:
                inliers.append(match)
    return inliers

def RANSAC(kps1, kps2, dmatch, max_iters=1000, inlier_ratio=0.9):
    best_inliers = []
    for i in range(max_iters):
        pts1, pts2 = select_random_points(kps1, kps2, dmatch, num_points=4)
        H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        inliers = find_inliers(H, kps1, kps2, dmatch, threshold= 15.0)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
        if len(best_inliers) > (inlier_ratio * len(dmatch)):
            break
    return best_inliers

def poisson_blend(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]])
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]])
    pts2_ = cv2.perspectiveTransform(pts2.reshape(-1, 1, 2), H).reshape(-1, 2)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0))
    [xmax, ymax] = np.int32(pts.max(axis=0))
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    result = cv2.warpPerspective(img1, Ht.dot(H), (xmax - xmin, ymax - ymin), flags=cv2.INTER_LINEAR)

    result_region = result[t[1]:h2 + t[1], t[0]:w2 + t[0]]
    
    if result_region.shape[0] != h2 or result_region.shape[1] != w2:
        result_region = result_region[:h2, :w2]

    result[t[1]:h2 + t[1], t[0]:w2 + t[0]] = img2

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setNum', default=1, choices=[1, 2, 3], help="Set number to run")
    args = parser.parse_args()

    ## Read images 
    images = [cv2.imread(f"Phase1/Data/Train/Set{args.setNum}/{i}.jpg") for i in range(1, 4)]
    im1 = images[0]
    im2 = images[1]
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    ## Detect corners in image
    corners1 = cv2.goodFeaturesToTrack(gray1, maxCorners=500, qualityLevel=.001, minDistance=10)
    corners1 = np.int0(corners1)
    corners1 = np.squeeze(corners1)  ## reshape from (maxCorners, 1, 2) to (maxCorners, 2)
    corners2 = cv2.goodFeaturesToTrack(gray2, maxCorners=500, qualityLevel=.001, minDistance=10)
    corners2 = np.int0(corners2)
    corners2 = np.squeeze(corners2)  ## reshape from (maxCorners, 1, 2) to (maxCorners, 2)
    plot_corners(im1.copy(), corners1, 'Phase1/im1_corners.jpg')
    plot_corners(im2.copy(), corners2, 'Phase1/im2_corners.jpg')
    
    ## Perform adaptive non-max supprression to find the "best" corners out of the corners found previously
    best_corners1 = ANMS(gray1, corners1, 150)
    best_corners2 = ANMS(gray2, corners2, 150)
    plot_corners(im1.copy(), best_corners1, 'Phase1/anms1.jpg')
    plot_corners(im2.copy(), best_corners2, 'Phase1/anms2.jpg')

    ## Find descriptors of the best corners which is a 8x8 patch around the corners 
    ## and then flattened to a 64 x 1 feature vector
    descriptors1 = getFeatureVector(gray1, best_corners1)  ## list of descriptors
    descriptors2 = getFeatureVector(gray2, best_corners2)  ## list of descriptors

    ## Measure similarity between descriptors to find keypoints, which are similar corners in each image
    kps1, kps2, dmatch = matchFeatures(descriptors1, descriptors2, best_corners1, best_corners2, 0.8)
    draw_matches(im1.copy(), im2.copy(), kps1, kps2, 'Phase1/matches.jpg')

    ## RANSAC to estimate homography between images
    inlier_matches = RANSAC(kps1, kps2, dmatch)
    in_kps1 = [kps1[inlier_matches[i].queryIdx] for i in range(len(inlier_matches))]
    in_kps2 = [kps2[inlier_matches[i].trainIdx] for i in range(len(inlier_matches))]
    draw_matches(im1.copy(), im2.copy(), in_kps1, in_kps2, 'Phase1/inlier_matches.jpg')
    H, _ = cv2.findHomography(np.asarray(in_kps1), np.asarray(in_kps2), method=0)

    ## Blend images with poisson blending
    blended_image = poisson_blend(im1.copy(), im2.copy(), H)
    save_image(blended_image, "Phase1/blended.jpg")

if __name__ == "__main__":
    main()

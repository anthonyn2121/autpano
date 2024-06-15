import numpy as np
import matplotlib.pyplot as plt
import cv2

def save_image(image, filename="image.jpg"):
    cv2.imwrite(filename, image)

def plot_corners(image, corners, filename='image.jpg'):
    for x, y in corners:
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    save_image(image, filename)

def draw_matches(img1, img2, kps1, kps2, filename):
    NumKps = len(kps1)
    MatchImage = np.concatenate((img1,img2), axis=1)
    for i in range(NumKps):
        x1, y1 = kps1[i][0], kps1[i][1]
        x2, y2 = kps2[i][0]+int(img1.shape[1]), kps2[i][1]
        cv2.line(MatchImage,(x1,y1),(x2,y2),(255,255,153),2)
        cv2.circle(MatchImage,(x1,y1),3,255,-1)
        cv2.circle(MatchImage,(x2,y2),3,255,-1)    
    save_image(MatchImage, filename)

def select_random_points(kps1, kps2, dmatch, num_points=4):
    sample = np.random.randint(0, len(dmatch)-1, size=num_points)
    pts1 = [kps1[dmatch[i].queryIdx] for i in sample]
    pts2 = [kps2[dmatch[i].trainIdx] for i in sample]
    return np.asarray(pts1), np.asarray(pts2)

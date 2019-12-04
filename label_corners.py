import cv2
import numpy as np
import sys
import pickle
import random
import os
from argparse import ArgumentParser
import json

'''
WIDTH: 79.5 cm
HEIGHT: 52.5cm

aspect ratio: 79.5/52.5

label_corners:
    - load in corner_samples/corner_labels.json
    - iterate through every image in board_images/
        - if image not already labeled (not in the json) and image
          is not already deskewed:
            - prompt the user to click on the corners in
                clockwise direction starting from top left
            - add labels to corner_labels.json
                -> 1x8 vector



'''

RATIO = 79.5 / 52.5
WIDTH = 1500
HEIGHT = int(WIDTH / RATIO) # 990 if width = 1500

# filename = 'IMG_4087.jpg'
# filename = 'IMG_4048_blank_deskewed.jpg'
# img_path = sys.argv[1]

# argparse = ArgumentParser(description='arg parser for cropping boards')
# argparse.add_argument('-i', '--image-path', type=str)
# args = argparse.parse_args()
# img_path = args.image_path
# path_split = os.path.split(img_path)
# filename = path_split[-1]
# img_dir = os.path.join(*path_split[:-1])
# image = cv2.imread(img_path)

# if 'deskewed' not in filename:
    # scale_percent = 25 # percent of original size
    # width = int(image.shape[1] * scale_percent / 100)
    # height = int(image.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # print(dim)

    # image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

mouse_coords = []

circle_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
color_idx = 0

tracks = pickle.load( open( "tracks.p", "rb" ) )

def create_point(x, y):
    global mouse_coords, image, circle_colors, color_idx, tracks
    color = circle_colors[color_idx]
    color_idx = (color_idx + 1) % len(circle_colors)

    cv2.circle(image,(x,y),2,color,-1)
    mouse_coords.append([x, y])
    cv2.imshow('image',image)

def delete_point():
    global mouse_coords, image, circle_colors, color_idx, tracks
    last_x, last_y = mouse_coords.pop()

    color_idx = (color_idx - 1) % len(circle_colors)

    cv2.circle(image,(last_x, last_y), 2, (0,0,0), -1)
    cv2.imshow('image',image)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
       # draw circle here (etc...)
       # cv2.circle(image, (x,y),100,(255,0,0),-1)
       print('x = %d, y = %d'%(x, y))
       create_point(x, y)

def open_and_resize(path):
    image = cv2.imread(path)
    scale_percent = 25 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    print(dim)

    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

# read in JSON file
json_file = 'corner_samples/corner_labels.json'
corner_labels = {}
if os.path.isfile(json_file):
    corner_labels = json.loads(open(json_file, 'r').read())

# read in image paths
image_paths = []
image_dir = 'board_images'

for file in os.listdir(image_dir):
    if 'deskewed' not in file and file not in corner_labels and not file.startswith('.'):
        image_paths.append(file)

image_idx = 0
# image = cv2.imread(os.path.join(image_dir, image_paths[image_idx]))
image = open_and_resize(os.path.join(image_dir, image_paths[image_idx]))
source_image = image.copy()

print(image_paths)

cv2.namedWindow('image')
cv2.setMouseCallback('image', on_mouse)
cv2.imshow('image',image)

while True:
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break

    elif k == ord('d'):
        # print(mouseX,mouseY)
        print("DELETE")
        delete_point()

    elif k == ord('n'):
        print(mouse_coords[-4:])
        corner_labels[image_paths[image_idx]] = mouse_coords[-4:]
        open(json_file, 'w').write(json.dumps(corner_labels))
        cv2.imwrite(os.path.join('corner_samples', image_paths[image_idx]), source_image)

        print("next image")
        image_idx += 1
        image = open_and_resize(os.path.join(image_dir, image_paths[image_idx]))
        source_image = image.copy()
        cv2.imshow('image',image)
        
        




cv2.destroyAllWindows()


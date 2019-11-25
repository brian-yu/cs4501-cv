import cv2
import numpy as np
import pickle
import math

'''
WIDTH: 79.5 cm
HEIGHT: 52.5cm

aspect ratio: 79.5/52.5
'''

tracks = pickle.load( open( "tracks.p", "rb" ) )

RATIO = 79.5 / 52.5
WIDTH = 1500
HEIGHT = int(WIDTH / RATIO) # 990 if width = 1500

TRAIN_WIDTH = 50
TRAIN_HEIGHT = 20


filename = 'IMG_4087_deskewed.jpg'
# filename = 'IMG_4048_blank_deskewed.jpg'
image = cv2.imread(filename)

if 'deskewed' not in filename:
    scale_percent = 25 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    print(dim)

    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)


image_idx = 0
def deskew(image, pts, target_width, target_height):
    global image_idx
    # print(pts)

    pts_src = np.array(pts)

    pts_dst = np.array(
        [[0, 0], [target_width-1, 0],
        [target_width-1, target_height-1], [0, target_height-1]])
    # top left, top right, bottom right, bottom left

    h, status = cv2.findHomography(pts_src, pts_dst)

    im_out = cv2.warpPerspective(image, h, (target_width, target_height))

    if image_idx == 27:
        print(pts)

    cv2.imwrite(f"samples/{image_idx}.jpg", im_out)
    image_idx += 1

    return im_out

def distance(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return ((y1-y2)**2 + (x1-x2)**2)

def reorder_points(box):

    # NEED TO ROTATE BOXES CORRECTLY.

    # distances = []

    # for i in range(len(box)):
    #     for j in range(i + 1, len(box)):
    #         distances.append((distance(box[i], box[j]), (i, j)))

    # distances.sort()

    # pt1, pt2 = box[distances[-1][1][0]], box[distances[-1][1][1]]
    # pt3, pt4 = box[distances[-2][1][0]], box[distances[-2][1][1]]
    # print(pt1, pt2)

    # return [pt1, pt4, pt2, pt3]

    sorted_x = sorted(box)

    # print(sorted_x)

    if sorted_x[0][1] < sorted_x[1][1]:
        top_left = sorted_x[0]
        bottom_left = sorted_x[1]
    else:
        top_left = sorted_x[1]
        bottom_left = sorted_x[0]

    if sorted_x[2][1] < sorted_x[3][1]:
        top_right = sorted_x[2]
        bottom_right = sorted_x[3]
    else:
        top_right = sorted_x[2]
        bottom_right = sorted_x[3]

    if distance(top_left, bottom_left) > distance(top_left, top_right):
        # return [bottom_right, top_right, bottom_left, top_left]
        return box

    # return [top_left, top_right, bottom_right, bottom_left]
    return box

# cv2.namedWindow('image')
# cv2.setMouseCallback('image', on_mouse)

import random

print(len(tracks))

for track in tracks:

    color = (random.randint(0, 255), random.randint(0, 255),
        random.randint(0, 255))

    for box in track:
        pts = np.array(box, np.int32)
        pts = pts.reshape((-1,1,2))

        deskew(image, reorder_points(box), TRAIN_WIDTH, TRAIN_HEIGHT)

        # cv2.polylines(image,[pts],True,color, 3)



# while True:
#     cv2.imshow('image',image)
#     k = cv2.waitKey(0) & 0xFF
#     if k == 27:
#         break
#     elif k == ord('c'):
#         # print(mouseX,mouseY)
#         print("CROP")
#         image = deskew(image)
#     elif k == ord('d'):
#         # print(mouseX,mouseY)
#         print("DELETE")
#         delete_point()

#     elif k == ord('s'):
#         # print(mouseX,mouseY)
#         print("saving track")
#         save_track()

        

# cv2.destroyAllWindows()



"""

"""
import cv2
import numpy as np
import pickle
import math
import sys
import os
import random

'''
WIDTH: 79.5 cm
HEIGHT: 52.5cm

aspect ratio: 79.5/52.5


images: 4168 4163 4143 4138 4134
'''

tracks = pickle.load( open( "tracks.p", "rb" ) )

RATIO = 79.5 / 52.5
WIDTH = 1500
HEIGHT = int(WIDTH / RATIO) # 990 if width = 1500

TRAIN_WIDTH = 100
TRAIN_HEIGHT = 100

img_path = sys.argv[1]
path_split = os.path.split(img_path)
filename = path_split[-1]
img_dir = os.path.join(*path_split[:-1])
image = cv2.imread(img_path)


def deskew(image, pts, target_width, target_height):
    # print(pts)

    pts_src = np.array(pts)

    pts_dst = np.array(
        [[0, 0], [target_width-1, 0],
        [target_width-1, target_height-1], [0, target_height-1]])
    # top left, top right, bottom right, bottom left

    h, status = cv2.findHomography(pts_src, pts_dst)

    im_out = cv2.warpPerspective(image, h, (target_width, target_height))

    return im_out


def extract_from_image(img_path):

    path_split = os.path.split(img_path)
    filename = path_split[-1]
    prefix = filename.split(".")[0]
    img_dir = os.path.join(*path_split[:-1])


    image = cv2.imread(img_path)
    corner_pts = json.loads(
        open('corner_samples/corner_labels.json', 'r').read())[filename]

    image = deskew(image, corner_pts, WIDTH, HEIGHT)


    image_idx = 0

    for track in tracks:

        color = (random.randint(0, 255), random.randint(0, 255),
            random.randint(0, 255))

        for box in track:
            pts = np.array(box, np.int32)
            pts = pts.reshape((-1,1,2))

            deskewed = deskew(image, box, TRAIN_WIDTH, TRAIN_HEIGHT)
            cv2.imwrite(f"samples/{prefix}-{image_idx}.jpg", deskewed)
            image_idx += 1



# track_idx = 0
# while True:
#     cv2.imshow('image',image)
#     k = cv2.waitKey(0) & 0xFF

#     if k == 27:
#         break
#     elif k == ord('n'):
#         if len(tracks[track_idx]) == 7:
#             print(tracks[track_idx])
#             print()
#             print(tracks[track_idx][:5])
#             print(tracks[track_idx][5:])
#         for box in tracks[track_idx]:
#             pts = np.array(box, np.int32)
#             pts = pts.reshape((-1,1,2))
            
#             new_img = image.copy()

#             cv2.polylines(new_img,[pts],True, (255, 255, 0) , 3)
#             image = new_img
#         track_idx += 1
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
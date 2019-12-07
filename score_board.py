import cv2
import numpy as np
import pickle
import sys
import os
from collections import Counter, defaultdict
import time

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image




RATIO = 79.5 / 52.5
WIDTH = 1500
HEIGHT = int(WIDTH / RATIO) # 990 if width = 1500

TRAIN_WIDTH = 100
TRAIN_HEIGHT = 100

def predict(image, model):

    label_names = ["red", "green", "blue", "black", "yellow", "none"]

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    input_img = preprocess(img).unsqueeze(0)

    output = model(input_img)

    pred = torch.argmax(output[0]).item()

    return label_names[pred]

def extract(image, pts, target_width, target_height):

    pts_src = np.array(pts)

    pts_dst = np.array(
        [[0, 0], [target_width-1, 0],
        [target_width-1, target_height-1], [0, target_height-1]])
    # top left, top right, bottom right, bottom left

    h, status = cv2.findHomography(pts_src, pts_dst)

    im_out = cv2.warpPerspective(image, h, (target_width, target_height))

    return im_out


# Accepts a cropped image
def score(image):
    tracks = pickle.load( open( "tracks.p", "rb" ) )
    print(f"{len(tracks)} tracks")

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)
    model.load_state_dict(torch.load("ticket_to_ride_model.pt", map_location='cpu'))
    model.eval()

    # labeled_boxes = []
    labeled_tracks = []
    score_map = {
        1: 1,
        2: 2,
        3: 4,
        4: 7,
        5: 10,
        6: 15
    }
    color_scores = defaultdict(int)

    print("Scoring...")
    start = time.time()
    for track in tracks:
        colors = Counter()
        for box in track:
            pts = np.array(box, np.int32).reshape((-1,1,2))

            box_img = extract(image, box, TRAIN_WIDTH, TRAIN_HEIGHT)

            pred_color = predict(box_img, model)
            # labeled_boxes.append((pts, pred_color))
            colors[pred_color] += 1

        color = colors.most_common(1)[0][0]

        # if top 2 are tied and one of them is none, use the other color.
        if len(colors) >= 2:
            top_2 = colors.most_common(2)
            if color == 'none' and top_2[0][1] == top_2[1][1]:
                color = top_2[1][0]

        if color != 'none':
            color_scores[color] += score_map[len(track)]

        labeled_tracks.append((track, color))

    print(f"Finished in {time.time() - start} seconds.")

    print("====== Scores ======")
    place = 0
    for color, score in sorted(color_scores.items(), key=lambda x: -x[1]):
        print(f"{place + 1}. {color.title()}: {score}")
        place += 1

    colors = {
        "red": (0,0,255),
        "green": (0,255,0),
        "blue": (255,0,0),
        "black": (0,0,0),
        "yellow": (0,255,255),
        "none": (255,255,255)
    }

    # for pts, color in labeled_boxes:
    #     cv2.polylines(image,[pts],True, colors[color], 3)

    for track, color in labeled_tracks:
        for box in track:
            pts = np.array(box, np.int32).reshape((-1,1,2))
            cv2.polylines(image,[pts],True, colors[color], 3)



    return image, color_scores




img_path = sys.argv[1]
image = cv2.imread(img_path)
h, w, _ = image.shape
target_width = 1000
image = cv2.resize(image, (target_width, int(target_width * (h/w))))
source_image = image.copy()

mouse_coords = []

def create_point(x, y):
    global mouse_coords, image

    if len(mouse_coords) >= 4:
        print("4 points already placed.")
        return

    cv2.circle(image,(x,y),3,(255,255,0),-1)
    mouse_coords.append([x, y])
    cv2.imshow('ResNet to Ride',image)

def delete_point():
    global mouse_coords, image

    if len(mouse_coords) == 0:
        print("No points to delete.")
        return

    last_x, last_y = mouse_coords.pop()

    cv2.circle(image,(last_x, last_y), 3, (0,0,0), -1)
    cv2.imshow('ResNet to Ride',image)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
       create_point(x, y)

h, w, _ = image.shape

image = cv2.rectangle(image, (30, h//2- 20), (w - 30, h//2+65), (0,0,0), -1)

image = cv2.putText(image,
    'Click on each corner of the board. Please try to be as accurate as possible.',
    (30, h//2), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255) , 2, cv2.LINE_AA)

image = cv2.putText(image,
    'Press \'d\' to undo a corner point',
    (30, h//2+30), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255) , 2, cv2.LINE_AA)

image = cv2.putText(image,
    'Press \'n\' after selecting corners to score the board.',
    (30, h//2+60), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255) , 2, cv2.LINE_AA) 

cv2.imshow('ResNet to Ride', image)

cv2.setMouseCallback('ResNet to Ride', on_mouse)

while True:
    cv2.imshow('ResNet to Ride',image)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break
    elif k == ord('n'):
        print("Cropping board.")
        
        pts = mouse_coords[-4:]
        sort_x = sorted(pts, key=lambda x: x[0])

        if sort_x[0][1] < sort_x[1][1]:
            top_left = sort_x[0]
            bottom_left = sort_x[1]
        else:
            top_left = sort_x[1]
            bottom_left = sort_x[0]

        if sort_x[-1][1] < sort_x[-2][1]:
            top_right = sort_x[-1]
            bottom_right = sort_x[-2]
        else:
            top_right = sort_x[-2]
            bottom_right = sort_x[-1]

        ordered_pts = [top_left, top_right, bottom_right, bottom_left]
        cropped = extract(source_image, ordered_pts, WIDTH, HEIGHT)
        image, scores = score(cropped)

        h, w, _ = image.shape
        start_x = 30
        start_y = h - 200

        image = cv2.rectangle(image, (start_x, start_y-30), (start_x+160, start_y+120), (0,0,0), -1)

        place = 0
        for color, score in sorted(scores.items(), key=lambda x: -x[1]):
            image = cv2.putText(image, f"{place + 1}. {color.title()}: {score}",
            (start_x, start_y + place * 25), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255) , 2,cv2.LINE_AA)
            place += 1


        cv2.imshow('ResNet to Ride', image)

    elif k == ord('d'):
        # print(mouseX,mouseY)
        print("Deleting point.")
        delete_point()


cv2.destroyAllWindows()

# cv2.waitKey(0)
# cv2.destroyAllWindows()